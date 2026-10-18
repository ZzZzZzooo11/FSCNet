import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
from functools import partial


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tensorboardX import SummaryWriter


import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ptflops import get_model_complexity_info
from medpy.metric.binary import hd95


from utils import DiceLoss
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator


def _worker_init_fn(worker_id, seed):

    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def calculate_binary_metrics(pred, target):


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


def calculate_hd95_with_empty_policy(pred, target):
    pred_nonempty = bool(np.any(pred))
    target_nonempty = bool(np.any(target))

    if pred_nonempty and target_nonempty:
        result = float(hd95(pred, target))
        if not np.isfinite(result):
            raise ValueError("HD95 returned a non-finite value for non-empty masks.")
        return result, 'normal'

    if not pred_nonempty and not target_nonempty:
        return 0.0, 'both_empty'

    height, width = pred.shape[-2:]
    penalty = float(np.hypot(height - 1, width - 1))
    if not pred_nonempty:
        return penalty, 'empty_prediction'
    return penalty, 'empty_target'


def draw_loss_curve(history, save_path):

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

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix.astype(int),
                                  display_labels=['Background', 'Nodule'])
    disp.plot(cmap=plt.cm.Blues, values_format='.0f')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()


def validate(model, valloader, ce_loss, dice_loss):
    model.eval()

    metrics_accum = {
        'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0,
        'specificity': 0.0, 'f1_score': 0.0, 'hd95': 0.0,
        'confusion': np.zeros((2, 2))
    }


    sample_scores = {'dice': [], 'iou': []}

    val_loss_total = 0
    hd95_count = 0
    hd95_status_counts = {
        'normal': 0,
        'empty_prediction': 0,
        'empty_target': 0,
        'both_empty': 0
    }
    total_samples = 0
    start_time = time.time()


    for sampled_batch in tqdm(valloader, desc="Validating", leave=False):
        image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

        with torch.no_grad():
            outputs = model(image_batch)


            l_ce = ce_loss(outputs, label_batch.long())
            l_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * l_ce + 0.5 * l_dice
            val_loss_total += loss.item() * image_batch.size(0)


            probs = F.softmax(outputs, dim=1)
            preds_batch = torch.argmax(probs, dim=1).cpu().numpy()
            labels_batch = label_batch.cpu().numpy()


            bs = image_batch.size(0)
            for i in range(bs):
                total_samples += 1
                pred_nodule = (preds_batch[i] == 1).astype(np.int32)
                target_nodule = (labels_batch[i] == 1).astype(np.int32)


                s_metric = calculate_binary_metrics(pred_nodule, target_nodule)


                sample_scores['dice'].append(s_metric['dice'])
                sample_scores['iou'].append(s_metric['iou'])


                for k in ['dice', 'iou', 'precision', 'recall', 'specificity', 'f1_score']:
                    metrics_accum[k] += s_metric[k]


                hd95_value, hd95_status = calculate_hd95_with_empty_policy(
                    pred_nodule, target_nodule
                )
                metrics_accum['hd95'] += hd95_value
                hd95_count += 1
                hd95_status_counts[hd95_status] += 1


                tp, fp, fn, tn = s_metric['confusion']
                metrics_accum['confusion'][0, 0] += tn
                metrics_accum['confusion'][0, 1] += fp
                metrics_accum['confusion'][1, 0] += fn
                metrics_accum['confusion'][1, 1] += tp

    if total_samples == 0: total_samples = 1

    avg_metrics = {k: v / total_samples for k, v in metrics_accum.items() if k != 'confusion' and k != 'hd95'}
    avg_metrics['hd95'] = metrics_accum['hd95'] / hd95_count if hd95_count > 0 else float('nan')
    avg_metrics['loss'] = val_loss_total / total_samples
    avg_metrics['confusion'] = metrics_accum['confusion']

    logging.info(
        "Validation HD95: total=%d, valid=%d, normal=%d, "
        "empty_prediction=%d, empty_target=%d, both_empty=%d; "
        "one-sided empty masks use the image-diagonal penalty in pixels.",
        total_samples, hd95_count,
        hd95_status_counts['normal'],
        hd95_status_counts['empty_prediction'],
        hd95_status_counts['empty_target'],
        hd95_status_counts['both_empty']
    )

    return avg_metrics, time.time() - start_time, metrics_accum['confusion'], sample_scores


def benchmark_inference_time(model, valloader, warmup_runs=30, measure_runs=100):


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


def save_efficiency_metrics(save_path, input_channels, img_size, params, macs,
                            inference_mean_ms, inference_std_ms, inference_runs,
                            training_time_hours, gpu_name):

    txt_path = os.path.join(save_path, 'efficiency_metrics.txt')
    csv_path = os.path.join(save_path, 'efficiency_metrics.csv')
    macs_text = str(macs) if macs is not None else 'N/A'

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"Input Shape: 1 x {input_channels} x {img_size} x {img_size}\n")
        f.write(f"Total Parameters: {params}\n")
        f.write(f"ptflops counted MACs: {macs_text}\n")
        f.write("Note: rFFT/irFFT and complex operations may not be counted; "
                "MACs are not complete FLOPs.\n")
        f.write(f"Inference Time Mean (ms/image): {inference_mean_ms:.6f}\n")
        f.write(f"Inference Time Std (ms/image): {inference_std_ms:.6f}\n")
        f.write(f"Inference Runs: {inference_runs}\n")
        f.write(f"Training Time (h, cumulative active): {training_time_hours:.6f}\n")

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('GPU,Input_Channels,Image_Size,Total_Parameters,ptflops_Counted_MACs,'
                'Inference_Time_Mean_ms,Inference_Time_Std_ms,Inference_Runs,'
                'Training_Time_h\n')
        safe_gpu_name = gpu_name.replace(',', ' ')
        f.write(f'{safe_gpu_name},{input_channels},{img_size},{params},{macs_text},'
                f'{inference_mean_ms:.6f},{inference_std_ms:.6f},{inference_runs},'
                f'{training_time_hours:.6f}\n')


def _unwrap_model(model):

    return model.module if isinstance(model, nn.DataParallel) else model


def _capture_rng_state():

    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):


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

                torch.cuda.set_rng_state(
                    restored_cuda_states[0],
                    device=torch.cuda.current_device()
                )

    except Exception as e:
        logging.warning(
            f"CUDA RNG state restore skipped: {e}"
        )


def _move_optimizer_state_to_device(optimizer, device):

    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def save_training_time_state(time_state_path, cumulative_seconds):


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

    if not os.path.isfile(time_state_path):
        return 0.0

    with open(time_state_path, 'r', encoding='utf-8') as f:
        value = f.read().strip()

    return float(value) if value else 0.0


def save_resume_checkpoint(checkpoint_path, model, optimizer, completed_epoch,
                           iter_num, best_metrics, best_confusion,
                           loss_history, val_history,
                           accumulated_training_seconds):


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


def trainer_synapse(args, model, snapshot_path):

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S', force=True)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"Training started at {start_time}")
    logging.info(f"Objective: 0.5*CE + 0.5*Dice | Optimizer: AdamW")


    default_checkpoint_path = os.path.join(snapshot_path, 'last_checkpoint.pth')
    configured_checkpoint_path = getattr(args, 'resume_checkpoint', '')
    checkpoint_path = configured_checkpoint_path or default_checkpoint_path
    time_state_path = os.path.join(snapshot_path, 'training_time_state.txt')
    auto_resume = bool(getattr(args, 'auto_resume', True))
    resume_available = auto_resume and os.path.isfile(checkpoint_path)


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


    real_macs, real_params = None, None


    input_channels = getattr(args, 'n_channels', 3)

    try:
        macs, _ = get_model_complexity_info(
            model, (input_channels, args.img_size, args.img_size),
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
        real_macs = macs
        logging.warning(
            "ptflops reports counted MACs; rFFT/irFFT and complex operations "
            "may not be included. Do not interpret this value as full FLOPs."
        )
    except Exception as e:
        logging.warning(f"MACs calculation skipped: {e}")


    real_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def format_num(n):
        return f"{n / 1e9:.2f} G" if n > 1e9 else (f"{n / 1e6:.2f} M" if n > 1e6 else f"{n:.2f}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    logging.info("-" * 60)
    logging.info(f"Hardware:         {gpu_name}")
    logging.info(f"Input Channels:   {input_channels}")
    logging.info(f"Input Resolution: {args.img_size} x {args.img_size}")
    logging.info(f"Total Parameters: {format_num(real_params)}")
    logging.info(f"Counted MACs:     {format_num(real_macs) if real_macs is not None else 'N/A'}")
    logging.info("Inference Time:   measured after training with batch size 1")
    logging.info("Training Time:    cumulative active training wall time across resumes")
    logging.info(
        f"Time state sync:  every "
        f"{int(getattr(args, 'time_sync_interval', 20))} iterations"
    )
    logging.info("-" * 60)


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


    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)


    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)


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


            l_ce = ce_loss(outputs, label_batch.long())
            l_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * l_ce + 0.5 * l_dice

            optimizer.zero_grad()
            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            schedule_step = epoch_num * len(trainloader) + batch_idx
            progress = min(max(schedule_step / max_iterations, 0.0), 1.0)
            lr_ = 1e-3 * (1.0 - progress) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = float(lr_)

            optimizer.step()


            iter_num += 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{l_dice.item():.4f}"})


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


                if iter_num % 100 == 0:
                    image = image_batch[0, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs_vis[0, ...] * 50, iter_num)


        draw_loss_curve(loss_history, snapshot_path)


        if (epoch_num + 1) % args.val_interval == 0:
            val_metrics, val_time, conf_matrix, sample_scores = validate(model, valloader, ce_loss, dice_loss)


            val_history['epoch'].append(epoch_num + 1)
            val_history['dice'].append(val_metrics['dice'])
            val_history['iou'].append(val_metrics['iou'])
            draw_val_curve(val_history, snapshot_path)


            for metric, value in val_metrics.items():
                if metric != 'confusion': writer.add_scalar(f'val/{metric}', value, epoch_num + 1)


            with open(val_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{epoch_num + 1}\t{val_metrics['dice']:.4f}\t{val_metrics['iou']:.4f}\t"
                        f"{val_metrics['precision']:.4f}\t{val_metrics['recall']:.4f}\t"
                        f"{val_metrics['specificity']:.4f}\t{val_metrics['hd95']:.4f}\n")

            logging.info(f"Epoch {epoch_num + 1} Val: Dice={val_metrics['dice']:.4f}, IoU={val_metrics['iou']:.4f}")


            if val_metrics['dice'] > best_metrics['dice']:
                best_metrics['dice'] = val_metrics['dice']
                best_metrics['epoch'] = epoch_num + 1
                best_confusion = conf_matrix


                torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))


                stats_path = os.path.join(snapshot_path, 'best_validation_scores.npz')
                np.savez(stats_path, dice=np.array(sample_scores['dice']), iou=np.array(sample_scores['iou']))
                logging.info(f"New Best Model! Sample-wise scores saved to {stats_path}")


        if (epoch_num + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(snapshot_path, f'epoch_{epoch_num + 1}.pth')
            )


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


    best_model_path = os.path.join(snapshot_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        try:
            best_state = torch.load(best_model_path, map_location=next(model.parameters()).device)
            model.load_state_dict(best_state)
            logging.info(f"Best model loaded for inference benchmark: {best_model_path}")
        except Exception as e:
            logging.warning(f"Best model loading skipped before inference benchmark: {e}")

    inference_warmup = int(getattr(args, 'inference_warmup', 30))
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
    logging.info(f"Counted MACs (ptflops): {format_num(real_macs) if real_macs is not None else 'N/A'}")
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
        macs=real_macs,
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
