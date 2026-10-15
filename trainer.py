import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
from functools import partial

# --- PyTorch 相关 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tensorboardX import SummaryWriter

# --- 绘图与计算工具 ---
import matplotlib

matplotlib.use('Agg')  # 防止在无图形界面的服务器上报错
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ptflops import get_model_complexity_info
from medpy.metric.binary import hd95

# --- 自定义模块 (请确保这些文件在你的目录下) ---
from utils import DiceLoss
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator


# ==============================================================================
# 1. 基础工具函数
# ==============================================================================

def _worker_init_fn(worker_id, seed):
    """设置 DataLoader 的随机种子，保证可复现性"""
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def calculate_binary_metrics(pred, target):
    """
    计算二分类指标：Dice, IoU, Precision, Recall, Specificity
    输入: pred (预测掩码 0/1), target (真实标签 0/1)
    """
    smooth = 1e-5
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    tp = np.sum((pred_flat == 1) & (target_flat == 1))
    tn = np.sum((pred_flat == 0) & (target_flat == 0))
    fp = np.sum((pred_flat == 1) & (target_flat == 0))
    fn = np.sum((pred_flat == 0) & (target_flat == 1))

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    specificity = tn / (tn + fp + smooth)
    f1_score = 2 * (precision * recall) / (precision + recall + smooth)
    dice = (2 * tp) / (2 * tp + fp + fn + smooth)
    iou = tp / (tp + fp + fn + smooth)

    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'dice': dice,
        'iou': iou,
        'confusion': [tp, fp, fn, tn]
    }


# ==============================================================================
# 2. 绘图与可视化函数
# ==============================================================================

def draw_loss_curve(history, save_path):
    """绘制并保存训练 Loss 曲线"""
    iters = history['iter']
    plt.figure(figsize=(10, 5))
    plt.plot(iters, history['loss'], label='Total Loss', color='red', linewidth=2)
    plt.plot(iters, history['ce'], label='CE Loss', color='blue', linestyle='--', alpha=0.6)
    plt.plot(iters, history['dice'], label='Dice Loss', color='green', linestyle='--', alpha=0.6)

    plt.title('Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'), bbox_inches='tight')
    plt.close()


def draw_val_curve(history, save_path):
    """绘制并保存验证集指标变化曲线"""
    epochs = history['epoch']
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['dice'], label='Dice', color='purple', marker='.')
    plt.plot(epochs, history['iou'], label='IoU', color='orange', marker='.')

    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_path, 'val_curve.png'), bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(conf_matrix, save_dir):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix.astype(int),
                                  display_labels=['Background', 'Nodule'])
    disp.plot(cmap=plt.cm.Blues, values_format='.0f')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()


# ==============================================================================
# 3. 验证逻辑 (包含 T 检验数据收集)
# ==============================================================================

def validate(model, valloader, ce_loss, dice_loss):
    model.eval()

    metrics_accum = {
        'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0,
        'specificity': 0.0, 'f1_score': 0.0, 'hd95': 0.0,
        'confusion': np.zeros((2, 2))
    }

    # [关键] 用于存储所有样本的指标，以便后续进行 T 检验 (Significant Test)
    sample_scores = {'dice': [], 'iou': []}

    val_loss_total = 0
    hd95_count = 0
    total_samples = 0
    start_time = time.time()

    # 使用 leave=False 让进度条跑完消失，保持 log 整洁
    for sampled_batch in tqdm(valloader, desc="Validating", leave=False):
        image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

        with torch.no_grad():
            outputs = model(image_batch)

            # --- 混合损失计算 (Code 1 逻辑) ---
            l_ce = ce_loss(outputs, label_batch.long())
            l_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * l_ce + 0.5 * l_dice
            val_loss_total += loss.item() * image_batch.size(0)

            # --- 预测结果处理 ---
            probs = F.softmax(outputs, dim=1)
            preds_batch = torch.argmax(probs, dim=1).cpu().numpy()
            labels_batch = label_batch.cpu().numpy()

            # --- 逐样本计算指标 ---
            bs = image_batch.size(0)
            for i in range(bs):
                total_samples += 1
                pred_nodule = (preds_batch[i] == 1).astype(np.int32)
                target_nodule = (labels_batch[i] == 1).astype(np.int32)

                # 计算单样本指标
                s_metric = calculate_binary_metrics(pred_nodule, target_nodule)

                # [关键] 收集单样本数据
                sample_scores['dice'].append(s_metric['dice'])
                sample_scores['iou'].append(s_metric['iou'])

                # 累加平均指标
                for k in ['dice', 'iou', 'precision', 'recall', 'specificity', 'f1_score']:
                    metrics_accum[k] += s_metric[k]

                # HD95 计算 (处理边界不存在的情况)
                if np.any(target_nodule):
                    try:
                        metrics_accum['hd95'] += hd95(pred_nodule, target_nodule)
                        hd95_count += 1
                    except RuntimeError:
                        pass

                # 混淆矩阵累加
                tp, fp, fn, tn = s_metric['confusion']
                metrics_accum['confusion'][0, 0] += tp;
                metrics_accum['confusion'][0, 1] += fp
                metrics_accum['confusion'][1, 0] += fn;
                metrics_accum['confusion'][1, 1] += tn

    if total_samples == 0: total_samples = 1

    avg_metrics = {k: v / total_samples for k, v in metrics_accum.items() if k != 'confusion' and k != 'hd95'}
    avg_metrics['hd95'] = metrics_accum['hd95'] / hd95_count if hd95_count > 0 else float('nan')
    avg_metrics['loss'] = val_loss_total / total_samples
    avg_metrics['confusion'] = metrics_accum['confusion']

    # 返回 sample_scores 用于 T 检验
    return avg_metrics, time.time() - start_time, metrics_accum['confusion'], sample_scores


# ==============================================================================
# 4. 推理速度测试
# ==============================================================================

def benchmark_inference_time(model, valloader, warmup_runs=20, measure_runs=100):
    """
    测量单张图像的模型前向推理时间（ms/image）。

    统计规则：
    1. batch size 使用 valloader 当前设置，建议设为 1；
    2. 仅统计 model(image) 的前向时间；
    3. 不包含数据读取、CPU->GPU 传输、Softmax、Argmax 和指标计算；
    4. CUDA 环境下先预热，并在计时前后执行同步；
    5. 返回平均值、标准差和实际测试次数。
    """
    model.eval()
    device = next(model.parameters()).device

    try:
        first_batch = next(iter(valloader))
    except StopIteration:
        logging.warning("Inference time skipped: validation loader is empty.")
        return float('nan'), float('nan'), 0

    warmup_image = first_batch['image'].to(device, non_blocking=True)

    with torch.inference_mode():
        for _ in range(max(0, warmup_runs)):
            _ = model(warmup_image)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    timings_ms = []

    with torch.inference_mode():
        while len(timings_ms) < measure_runs:
            measured_this_round = 0

            for sampled_batch in valloader:
                image_batch = sampled_batch['image'].to(device, non_blocking=True)
                batch_size = image_batch.size(0)

                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starter.record()
                    _ = model(image_batch)
                    ender.record()
                    torch.cuda.synchronize(device)
                    elapsed_ms = starter.elapsed_time(ender)
                else:
                    start = time.perf_counter()
                    _ = model(image_batch)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0

                timings_ms.append(elapsed_ms / max(batch_size, 1))
                measured_this_round += 1

                if len(timings_ms) >= measure_runs:
                    break

            if measured_this_round == 0:
                break

    if not timings_ms:
        return float('nan'), float('nan'), 0

    timings_ms = np.asarray(timings_ms, dtype=np.float64)
    return float(timings_ms.mean()), float(timings_ms.std(ddof=0)), int(timings_ms.size)


def save_efficiency_metrics(save_path, input_channels, img_size, params, flops,
                            inference_mean_ms, inference_std_ms, inference_runs,
                            training_time_hours, gpu_name):
    """将模型复杂度、推理时间和训练时间同时保存为TXT与CSV。"""
    txt_path = os.path.join(save_path, 'efficiency_metrics.txt')
    csv_path = os.path.join(save_path, 'efficiency_metrics.csv')
    flops_text = str(flops) if flops is not None else 'N/A'

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"Input Shape: 1 x {input_channels} x {img_size} x {img_size}\n")
        f.write(f"Total Parameters: {params}\n")
        f.write(f"Total FLOPs: {flops_text}\n")
        f.write(f"Inference Time Mean (ms/image): {inference_mean_ms:.6f}\n")
        f.write(f"Inference Time Std (ms/image): {inference_std_ms:.6f}\n")
        f.write(f"Inference Runs: {inference_runs}\n")
        f.write(f"Training Time (h, cumulative active): {training_time_hours:.6f}\n")

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('GPU,Input_Channels,Image_Size,Total_Parameters,Total_FLOPs,'
                'Inference_Time_Mean_ms,Inference_Time_Std_ms,Inference_Runs,'
                'Training_Time_h\n')
        safe_gpu_name = gpu_name.replace(',', ' ')
        f.write(f'{safe_gpu_name},{input_channels},{img_size},{params},{flops_text},'
                f'{inference_mean_ms:.6f},{inference_std_ms:.6f},{inference_runs},'
                f'{training_time_hours:.6f}\n')


# ==============================================================================
# 5. 断点续训工具
# ==============================================================================

def _unwrap_model(model):
    """兼容普通模型与 nn.DataParallel，统一访问原始模型。"""
    return model.module if isinstance(model, nn.DataParallel) else model


def _capture_rng_state():
    """保存随机数生成器状态，使按 epoch 恢复后的训练尽可能可复现。"""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    """
    恢复随机数状态。

    强制将PyTorch RNG状态转换为CPU uint8 Tensor，
    避免检查点通过map_location加载到GPU后出现：
    RNG state must be a torch.ByteTensor。
    """
    if not state:
        return

    try:
        python_state = state.get('python')
        if python_state is not None:
            random.setstate(python_state)
    except Exception as e:
        logging.warning(
            f"Python RNG state restore skipped: {e}"
        )

    try:
        numpy_state = state.get('numpy')
        if numpy_state is not None:
            np.random.set_state(numpy_state)
    except Exception as e:
        logging.warning(
            f"NumPy RNG state restore skipped: {e}"
        )

    try:
        torch_state = state.get('torch')
        if torch_state is not None:
            if not torch.is_tensor(torch_state):
                torch_state = torch.tensor(
                    torch_state,
                    dtype=torch.uint8
                )
            else:
                torch_state = torch_state.detach().cpu().to(
                    dtype=torch.uint8
                )

            torch.set_rng_state(torch_state)
    except Exception as e:
        logging.warning(
            f"PyTorch CPU RNG state restore skipped: {e}"
        )

    try:
        cuda_states = state.get('cuda')

        if torch.cuda.is_available() and cuda_states is not None:
            restored_cuda_states = []

            for cuda_state in cuda_states:
                if not torch.is_tensor(cuda_state):
                    cuda_state = torch.tensor(
                        cuda_state,
                        dtype=torch.uint8
                    )
                else:
                    cuda_state = cuda_state.detach().cpu().to(
                        dtype=torch.uint8
                    )

                restored_cuda_states.append(cuda_state)

            device_count = torch.cuda.device_count()

            if len(restored_cuda_states) == device_count:
                torch.cuda.set_rng_state_all(
                    restored_cuda_states
                )
            elif len(restored_cuda_states) > 0:
                # GPU数量发生变化时，至少恢复当前GPU的状态
                torch.cuda.set_rng_state(
                    restored_cuda_states[0],
                    device=torch.cuda.current_device()
                )

    except Exception as e:
        logging.warning(
            f"CUDA RNG state restore skipped: {e}"
        )


def _move_optimizer_state_to_device(optimizer, device):
    """确保恢复后的优化器状态位于模型所在设备。"""
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def save_training_time_state(time_state_path, cumulative_seconds):
    """
    保存累计训练时间。

    Windows下若文件被杀毒软件、同步程序或其他进程短暂占用，
    自动重试；最终仍失败时只记录警告，不中断模型训练。
    """
    temporary_path = (
        f"{time_state_path}.{os.getpid()}.tmp"
    )

    for attempt in range(5):
        try:
            with open(
                temporary_path,
                'w',
                encoding='utf-8'
            ) as f:
                f.write(
                    f"{float(cumulative_seconds):.9f}\n"
                )
                f.flush()
                os.fsync(f.fileno())

            os.replace(
                temporary_path,
                time_state_path
            )
            return True

        except PermissionError as e:
            if attempt < 4:
                time.sleep(0.2 * (attempt + 1))
            else:
                logging.warning(
                    "Training time state could not be saved "
                    f"after 5 attempts: {e}. "
                    "Training will continue."
                )

        except OSError as e:
            logging.warning(
                "Training time state save failed: "
                f"{e}. Training will continue."
            )
            break

    try:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    except OSError:
        pass

    return False


def load_training_time_state(time_state_path):
    """读取累计有效训练时间；文件不存在时返回0。"""
    if not os.path.isfile(time_state_path):
        return 0.0

    with open(time_state_path, 'r', encoding='utf-8') as f:
        value = f.read().strip()

    return float(value) if value else 0.0


def save_resume_checkpoint(checkpoint_path, model, optimizer, completed_epoch,
                           iter_num, best_metrics, best_confusion,
                           loss_history, val_history,
                           accumulated_training_seconds):
    """
    原子方式保存完整训练状态。

    completed_epoch 为刚刚完整训练结束的 epoch 的 0-based 索引；
    恢复时从 completed_epoch + 1 开始。
    """
    checkpoint = {
        'format_version': 1,
        'model_state_dict': _unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'completed_epoch': int(completed_epoch),
        'iter_num': int(iter_num),
        'best_metrics': best_metrics,
        'best_confusion': best_confusion,
        'loss_history': loss_history,
        'val_history': val_history,
        'accumulated_training_seconds': float(accumulated_training_seconds),
        'rng_state': _capture_rng_state()
    }

    temporary_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def load_resume_checkpoint(checkpoint_path, model, optimizer):
    """加载完整检查点并返回恢复训练所需状态。"""
    device = next(_unwrap_model(model).parameters()).device

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu'
        )

    required_keys = {
        'model_state_dict',
        'optimizer_state_dict',
        'completed_epoch',
        'iter_num'
    }
    missing_keys = required_keys.difference(checkpoint.keys())
    if missing_keys:
        raise KeyError(
            f"检查点缺少必要字段：{sorted(missing_keys)}。"
            "该文件可能只是普通模型权重，不能用于完整断点续训。"
        )

    _unwrap_model(model).load_state_dict(
        checkpoint['model_state_dict'],
        strict=True
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    _move_optimizer_state_to_device(optimizer, device)
    _restore_rng_state(checkpoint.get('rng_state'))

    return {
        'start_epoch': int(checkpoint['completed_epoch']) + 1,
        'iter_num': int(checkpoint['iter_num']),
        'best_metrics': checkpoint.get(
            'best_metrics',
            {'dice': 0.0, 'epoch': 0}
        ),
        'best_confusion': checkpoint.get('best_confusion'),
        'loss_history': checkpoint.get(
            'loss_history',
            {'iter': [], 'loss': [], 'ce': [], 'dice': []}
        ),
        'val_history': checkpoint.get(
            'val_history',
            {'epoch': [], 'dice': [], 'iou': []}
        ),
        'accumulated_training_seconds': float(
            checkpoint.get('accumulated_training_seconds', 0.0)
        )
    }



# ==============================================================================
# 6. 主训练函数 (Trainer)
# ==============================================================================

def trainer_synapse(args, model, snapshot_path):
    # --- 1. 环境初始化 ---
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 配置 Logger
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S', force=True)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"Training started at {start_time}")
    logging.info(f"Objective: 0.5*CE + 0.5*Dice | Optimizer: AdamW")

    # 自动断点续训：默认检查 snapshot_path/last_checkpoint.pth。
    # 可通过 args.auto_resume=False 禁用，或通过 args.resume_checkpoint 指定其他路径。
    default_checkpoint_path = os.path.join(snapshot_path, 'last_checkpoint.pth')
    configured_checkpoint_path = getattr(args, 'resume_checkpoint', '')
    checkpoint_path = configured_checkpoint_path or default_checkpoint_path
    time_state_path = os.path.join(snapshot_path, 'training_time_state.txt')
    auto_resume = bool(getattr(args, 'auto_resume', True))
    resume_available = auto_resume and os.path.isfile(checkpoint_path)

    # 配置 TensorBoard 和文本日志。恢复训练时不覆盖已有日志。
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    loss_log_path = os.path.join(snapshot_path, "loss_log.txt")
    val_log_path = os.path.join(snapshot_path, "val_log.txt")

    log_mode = 'a' if resume_available else 'w'
    with open(loss_log_path, log_mode, encoding='utf-8') as f:
        if log_mode == 'w' or os.path.getsize(loss_log_path) == 0:
            f.write("Epoch\tIteration\tTotal_Loss\tCE_Loss\tDice_Loss\n")
    with open(val_log_path, log_mode, encoding='utf-8') as f:
        if log_mode == 'w' or os.path.getsize(val_log_path) == 0:
            f.write("Epoch\tDice\tIoU\tPrecision\tRecall\tSpecificity\tHD95\n")

    # --- 2. 模型复杂度计算 (修复版：自动适配通道数) ---
    real_flops, real_params = None, None

    # 尝试从 args 获取通道数，如果没有则默认为 1 (医学图像通常为1)
    input_channels = getattr(args, 'n_channels', 3)

    def input_constructor(input_res):
        # 构造虚拟输入: (Batch=1, Channels, H, W)
        return torch.ones(()).new_empty((1, input_channels, args.img_size, args.img_size)).cuda()

    try:
        flops, params = get_model_complexity_info(model, (input_channels, args.img_size, args.img_size),
                                                  as_strings=False, print_per_layer_stat=False,
                                                  verbose=False, input_constructor=input_constructor)
        real_flops = flops
    except Exception as e:
        logging.warning(f"⚠️ FLOPs calculation skipped: {e}")

    # 无论 FLOPs 是否成功，手动计算参数量
    real_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def format_num(n):
        return f"{n / 1e9:.2f} G" if n > 1e9 else (f"{n / 1e6:.2f} M" if n > 1e6 else f"{n:.2f}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    logging.info("-" * 60)
    logging.info(f"Hardware:         {gpu_name}")
    logging.info(f"Input Channels:   {input_channels}")
    logging.info(f"Input Resolution: {args.img_size} x {args.img_size}")
    logging.info(f"Total Parameters: {format_num(real_params)}")
    logging.info(f"Total FLOPs:      {format_num(real_flops) if real_flops is not None else 'N/A'}")
    logging.info("Inference Time:   measured after training with batch size 1")
    logging.info("Training Time:    cumulative active training wall time across resumes")
    logging.info(
        f"Time state sync:  every "
        f"{int(getattr(args, 'time_sync_interval', 20))} iterations"
    )
    logging.info("-" * 60)

    # --- 3. 数据加载 ---
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    init_fn = partial(_worker_init_fn, seed=args.seed)
    trainloader = DataLoader(db_train, batch_size=args.batch_size * args.n_gpu, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=init_fn)

    db_val = Synapse_dataset(base_dir='data/isic2018/val', list_dir='lists/isic2018_list', split="val",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, worker_init_fn=init_fn)

    logging.info(f"Train Set: {len(db_train)} | Val Set: {len(db_val)}")

    if args.n_gpu > 1: model = nn.DataParallel(model)

    # --- 4. 优化器与损失函数 ---
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    # 使用 AdamW 优化器 (LR=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # --- 5. 训练状态初始化与断点恢复 ---
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(trainloader)
    start_epoch = 0
    loss_history = {'iter': [], 'loss': [], 'ce': [], 'dice': []}
    val_history = {'epoch': [], 'dice': [], 'iou': []}
    best_metrics = {'dice': 0.0, 'epoch': 0}
    best_confusion = None
    accumulated_training_seconds = 0.0

    if resume_available:
        try:
            restored = load_resume_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer
            )
            start_epoch = restored['start_epoch']
            iter_num = restored['iter_num']
            best_metrics = restored['best_metrics']
            best_confusion = restored['best_confusion']
            loss_history = restored['loss_history']
            val_history = restored['val_history']
            accumulated_training_seconds = restored[
                'accumulated_training_seconds'
            ]

            # 时间状态文件会比 epoch 检查点更频繁地更新。
            # 取两者较大值，可保留断电前已运行但尚未完成的部分训练时间。
            try:
                persisted_time_seconds = load_training_time_state(
                    time_state_path
                )
                accumulated_training_seconds = max(
                    accumulated_training_seconds,
                    persisted_time_seconds
                )
            except Exception as e:
                logging.warning(
                    f"Training time state could not be read; "
                    f"checkpoint time will be used: {e}"
                )

            logging.info("=" * 70)
            logging.info(f"Checkpoint restored: {checkpoint_path}")
            logging.info(
                f"Resume from epoch {start_epoch + 1}/{max_epoch}; "
                f"restored iteration={iter_num}"
            )
            logging.info(
                f"Previously accumulated active training time: "
                f"{accumulated_training_seconds / 3600.0:.4f} h"
            )
            logging.info("=" * 70)
        except Exception as e:
            logging.exception(
                f"Checkpoint loading failed; training will stop to avoid "
                f"silently restarting from scratch: {e}"
            )
            writer.close()
            raise

    if start_epoch >= max_epoch:
        logging.info(
            f"Checkpoint has already completed {start_epoch} epochs, "
            f"which is not less than max_epochs={max_epoch}. "
            "No additional training epoch will be executed."
        )

    # 当前进程的训练计时。最终结果会加上检查点中保存的历史累计时间。
    current_session_training_start = time.perf_counter()
    trained_any_epoch = False
    time_sync_interval = max(
        1,
        int(getattr(args, 'time_sync_interval', 20))
    )

    for epoch_num in range(start_epoch, max_epoch):
        trained_any_epoch = True
        model.train()
        pbar = tqdm(trainloader, ncols=100, desc=f"Epoch [{epoch_num + 1}/{max_epoch}]")

        for batch_idx, sampled_batch in enumerate(pbar):
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            outputs = model(image_batch)

            # --- 损失计算 (Code 1 混合损失) ---
            l_ce = ce_loss(outputs, label_batch.long())
            l_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * l_ce + 0.5 * l_dice

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Poly 学习率：按 epoch 和当前 batch 计算训练进度。
            # checkpoint 恢复的 iter_num 仍保留用于日志、TensorBoard 和保存，
            # 但不作为学习率调度的步数，避免计数偏移导致提前归零或产生复数。
            schedule_step = epoch_num * len(trainloader) + batch_idx
            progress = min(max(schedule_step / max_iterations, 0.0), 1.0)
            lr_ = 1e-4 * (1.0 - progress) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = float(lr_)

            optimizer.step()

            # 记录
            iter_num += 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{l_dice.item():.4f}"})

            # 周期性保存累计有效训练时间。若突然断电，最多丢失
            # time_sync_interval - 1 个 iteration 的计时。
            if iter_num % time_sync_interval == 0:
                active_seconds_now = (
                    accumulated_training_seconds
                    + time.perf_counter()
                    - current_session_training_start
                )
                save_training_time_state(
                    time_state_path,
                    active_seconds_now
                )

            if iter_num % 20 == 0:
                loss_history['iter'].append(iter_num)
                loss_history['loss'].append(loss.item())
                loss_history['ce'].append(l_ce.item())
                loss_history['dice'].append(l_dice.item())
                with open(loss_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch_num + 1}\t{iter_num}\t{loss.item():.6f}\t{l_ce.item():.6f}\t{l_dice.item():.6f}\n")

                # 可视化图片记录 (每100个iter)
                if iter_num % 100 == 0:
                    image = image_batch[0, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs_vis[0, ...] * 50, iter_num)

        # 绘制 Loss 曲线图
        draw_loss_curve(loss_history, snapshot_path)

        # --- 验证环节 ---
        if (epoch_num + 1) % args.val_interval == 0:
            val_metrics, val_time, conf_matrix, sample_scores = validate(model, valloader, ce_loss, dice_loss)

            # 记录验证曲线
            val_history['epoch'].append(epoch_num + 1)
            val_history['dice'].append(val_metrics['dice'])
            val_history['iou'].append(val_metrics['iou'])
            draw_val_curve(val_history, snapshot_path)

            # TensorBoard 记录
            for metric, value in val_metrics.items():
                if metric != 'confusion': writer.add_scalar(f'val/{metric}', value, epoch_num + 1)

            # Txt 记录
            with open(val_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{epoch_num + 1}\t{val_metrics['dice']:.4f}\t{val_metrics['iou']:.4f}\t"
                        f"{val_metrics['precision']:.4f}\t{val_metrics['recall']:.4f}\t"
                        f"{val_metrics['specificity']:.4f}\t{val_metrics['hd95']:.4f}\n")

            logging.info(f"Epoch {epoch_num + 1} Val: Dice={val_metrics['dice']:.4f}, IoU={val_metrics['iou']:.4f}")

            # 保存最佳模型
            if val_metrics['dice'] > best_metrics['dice']:
                best_metrics['dice'] = val_metrics['dice']
                best_metrics['epoch'] = epoch_num + 1
                best_confusion = conf_matrix

                # 1. 保存模型权重
                torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))

                # 2. 【关键】保存 T 检验用的逐样本数据
                stats_path = os.path.join(snapshot_path, 'best_validation_scores.npz')
                np.savez(stats_path, dice=np.array(sample_scores['dice']), iou=np.array(sample_scores['iou']))
                logging.info(f"New Best Model! Sample-wise scores saved to {stats_path}")

        # 定期保存普通模型权重
        if (epoch_num + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(snapshot_path, f'epoch_{epoch_num + 1}.pth')
            )

        # 每个完整 epoch 结束后保存可恢复训练的完整检查点。
        # 若在一个 epoch 中途断电，会从上一个完整 epoch 重新开始。
        cumulative_seconds_at_checkpoint = (
            accumulated_training_seconds
            + time.perf_counter()
            - current_session_training_start
        )
        save_resume_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            completed_epoch=epoch_num,
            iter_num=iter_num,
            best_metrics=best_metrics,
            best_confusion=best_confusion,
            loss_history=loss_history,
            val_history=val_history,
            accumulated_training_seconds=cumulative_seconds_at_checkpoint
        )
        save_training_time_state(
            time_state_path,
            cumulative_seconds_at_checkpoint
        )
        logging.info(
            f"Resume checkpoint saved: {checkpoint_path} "
            f"(completed epoch {epoch_num + 1}, "
            f"active training time "
            f"{cumulative_seconds_at_checkpoint / 3600.0:.4f} h)"
        )

    # --------------------------------------------------------------------------
    # 训练时间与推理时间统计
    # --------------------------------------------------------------------------
    if trained_any_epoch:
        training_time_seconds = (
            accumulated_training_seconds
            + time.perf_counter()
            - current_session_training_start
        )
    else:
        training_time_seconds = accumulated_training_seconds

    training_time_hours = training_time_seconds / 3600.0
    save_training_time_state(
        time_state_path,
        training_time_seconds
    )

    # 优先加载最佳验证权重，再进行统一的推理速度测试。
    best_model_path = os.path.join(snapshot_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        try:
            best_state = torch.load(best_model_path, map_location=next(model.parameters()).device)
            model.load_state_dict(best_state)
            logging.info(f"Best model loaded for inference benchmark: {best_model_path}")
        except Exception as e:
            logging.warning(f"Best model loading skipped before inference benchmark: {e}")

    inference_warmup = int(getattr(args, 'inference_warmup', 20))
    inference_runs = int(getattr(args, 'inference_runs', 100))
    inference_mean_ms, inference_std_ms, actual_inference_runs = benchmark_inference_time(
        model=model,
        valloader=valloader,
        warmup_runs=inference_warmup,
        measure_runs=inference_runs
    )

    logging.info("=" * 60)
    logging.info("MODEL EFFICIENCY SUMMARY")
    logging.info(f"Hardware:              {gpu_name}")
    logging.info(f"Total Parameters:      {format_num(real_params)}")
    logging.info(f"Total FLOPs:           {format_num(real_flops) if real_flops is not None else 'N/A'}")
    logging.info(
        f"Inference Time (ms):   {inference_mean_ms:.3f} ± {inference_std_ms:.3f} "
        f"(batch=1, n={actual_inference_runs})"
    )
    logging.info(f"Training Time (h):     {training_time_hours:.4f}")
    logging.info("=" * 60)

    save_efficiency_metrics(
        save_path=snapshot_path,
        input_channels=input_channels,
        img_size=args.img_size,
        params=real_params,
        flops=real_flops,
        inference_mean_ms=inference_mean_ms,
        inference_std_ms=inference_std_ms,
        inference_runs=actual_inference_runs,
        training_time_hours=training_time_hours,
        gpu_name=gpu_name
    )

    logging.info("Training Finished.")
    if best_confusion is not None:
        plot_confusion_matrix(best_confusion, snapshot_path)
    writer.close()
    return "Training Finished!"
