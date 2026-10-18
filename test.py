import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.FSC_Net import Model
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

try:
    from medpy.metric.binary import hd95 as medpy_hd95
except ImportError:
    medpy_hd95 = None

METRIC_NAMES = ("dice", "iou", "precision", "recall", "specificity", "f1_score")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FSC-Net from saved model weights")
    parser.add_argument("--weights", type=str, default="logs/FSC-Net/FSC-Net/best_model.pth",
                        help="best_model.pth, epoch_*.pth, or last_checkpoint.pth")
    parser.add_argument("--test_root", type=str, default="data/isic2018/test",
                        help="Test dataset base_dir, matching Synapse_dataset")
    parser.add_argument("--list_dir", type=str, default="lists/isic2018_list")
    parser.add_argument("--split", type=str, default="test",
                        help="Requires dataset_synapse.py to support the test split")
    parser.add_argument("--save_dir", type=str, default="logs/FSC-Net/FSC-Net/test_results")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--base_c", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--transform", choices=("same_as_val", "deterministic"),
                        default="same_as_val",
                        help="same_as_val uses the trainer's RandomGenerator; deterministic does geometric resizing only")
    parser.add_argument("--no_save_preds", action="store_true", help="Do not save 0/255 PNG predictions")
    parser.add_argument("--no_benchmark", action="store_true", help="Skip isolated forward-pass timing")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--benchmark_runs", type=int, default=100)
    parser.add_argument("--trust_checkpoint", action="store_true",
                        help="Allow loading a trusted legacy/full checkpoint requiring pickle deserialization")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DeterministicTestTransform:

    def __init__(self, output_size, expected_channels=3):
        self.output_size = tuple(output_size)
        self.expected_channels = expected_channels

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if torch.is_tensor(label):
            label = label.detach().cpu().numpy()
        image = np.asarray(image)
        label = np.asarray(label)

        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        elif image.ndim == 3:
            if image.shape[0] == self.expected_channels:
                pass
            elif image.shape[-1] == self.expected_channels:
                image = np.moveaxis(image, -1, 0)
            else:
                raise ValueError(f"Cannot infer image channel axis: {image.shape}")
        else:
            raise ValueError(f"Expected image with 2 or 3 dimensions: {image.shape}")

        if label.ndim == 3:
            if label.shape[0] == 1:
                label = label[0]
            elif label.shape[-1] == 1:
                label = label[..., 0]
            else:
                raise ValueError(f"Expected single-channel label: {label.shape}")
        if label.ndim != 2:
            raise ValueError(f"Expected H,W label: {label.shape}")

        _, h, w = image.shape
        if label.shape != (h, w):
            raise ValueError(f"Image spatial shape {(h, w)} != label shape {label.shape}")
        target_h, target_w = self.output_size
        if (h, w) != (target_h, target_w):
            image = zoom(image, (1, target_h / h, target_w / w), order=3)
            label = zoom(label, (target_h / h, target_w / w), order=0)

        result = dict(sample)
        result["image"] = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        result["label"] = torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))
        return result


def load_model_weights(model, path, trust_checkpoint=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path.resolve()}")

    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        if not trust_checkpoint:
            raise RuntimeError(
                "Safe checkpoint loading failed. If this is YOUR OWN trusted "
                "last_checkpoint.pth, add --trust_checkpoint. Otherwise use "
                "best_model.pth. Original error: " + str(exc)
            ) from exc
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a model state_dict or checkpoint dict")
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        source = "model_state_dict in full checkpoint"
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state = checkpoint["state_dict"]
        source = "state_dict in checkpoint"
    else:
        state = checkpoint
        source = "plain state_dict"
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a valid model state_dict")
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: tensor for key, tensor in state.items()}
        source += ", removed DataParallel module. prefix"
    model.load_state_dict(state, strict=True)
    return source


def calculate_binary_metrics(pred, target):
    pred, target = pred.astype(bool), target.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    smooth = 1e-5
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + smooth),
        "iou": tp / (tp + fp + fn + smooth),
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp + smooth),
        "f1_score": 2 * precision * recall / (precision + recall + smooth),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def calculate_hd95(pred, target):
    pred_nonempty = bool(np.any(pred))
    target_nonempty = bool(np.any(target))

    if pred_nonempty and target_nonempty:
        if medpy_hd95 is None:
            raise ImportError("HD95 calculation requires medpy: pip install medpy")
        value = float(medpy_hd95(pred.astype(bool), target.astype(bool)))
        if not np.isfinite(value):
            raise ValueError("HD95 returned a non-finite value for non-empty masks")
        return value, "normal"

    if not pred_nonempty and not target_nonempty:
        return 0.0, "both_empty"

    height, width = pred.shape[-2:]
    penalty = float(np.hypot(height - 1, width - 1))
    if not pred_nonempty:
        return penalty, "empty_prediction"
    return penalty, "empty_target"


def case_names_for_batch(batch, batch_size, first_index, sample_list=None):
    raw = batch.get("case_name", batch.get("name", batch.get("id")))
    if raw is None:
        if sample_list is not None:
            selected = sample_list[first_index:first_index + batch_size]
            if len(selected) == batch_size:
                return [str(name).strip() for name in selected]
        return [f"sample_{first_index + j:05d}" for j in range(batch_size)]
    if isinstance(raw, str):
        raw = [raw]
    if torch.is_tensor(raw):
        raw = raw.cpu().tolist()
    if isinstance(raw, (tuple, list)) and len(raw) == batch_size:
        return [str(x) for x in raw]
    return [f"sample_{first_index + j:05d}" for j in range(batch_size)]


def safe_filename(name, index):
    stem = Path(str(name).replace("\\", "/")).stem
    stem = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]+", "_", stem)
    return f"{index:05d}_{stem or 'sample'}"


def inference_benchmark(model, sample, device, warmup=30, runs=100):
    if runs <= 0:
        return None
    model.eval()
    sample = sample[:1].to(device, non_blocking=True)
    times = []
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(runs):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(sample)
                end_event.record()
                end_event.synchronize()
                ms = start_event.elapsed_time(end_event)
            else:
                start = time.perf_counter()
                _ = model(sample)
                ms = (time.perf_counter() - start) * 1000.0
            times.append(float(ms))
    return {
        "mean_ms_per_image": float(np.mean(times)),
        "std_ms_per_image": float(np.std(times)),
        "measured_runs": len(times),
        "warmup_runs": max(0, warmup),
        "batch_size": 1,
        "only_model_forward": True,
    }


def run_test(args):
    if args.num_classes != 2:
        raise ValueError("This script matches your binary-class trainer; use --num_classes 2")
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.transform == "same_as_val":
        transform = RandomGenerator(output_size=[args.img_size, args.img_size])
        print("NOTICE: using RandomGenerator as in the validation loader; "
              "verify that upstream image normalization matches training.")
    else:
        transform = DeterministicTestTransform([args.img_size, args.img_size], args.img_channels)
        print("NOTICE: deterministic geometric resizing selected; confirm image normalization "
              "matches your dataset_synapse.py / RandomGenerator implementation.")

    dataset = Synapse_dataset(base_dir=args.test_root, list_dir=args.list_dir,
                              split=args.split, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError("Empty test dataset: verify --test_root, --list_dir, and --split")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    net = Model(img_channels=args.img_channels,
                       n_classes=args.num_classes, base_c=args.base_c)
    source = load_model_weights(net, args.weights, args.trust_checkpoint)
    net.to(device).eval()
    print(f"Device: {device} | Test cases: {len(dataset)} | Weights: {args.weights}")
    print(f"Checkpoint type: {source}")

    pred_dir = output_dir / "predictions"
    if not args.no_save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    first_benchmark_sample = None
    confusion = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    start = time.perf_counter()
    for batch in tqdm(loader, desc="Testing FSC-Net"):
        images = batch["image"]
        labels = batch["label"]
        if images.ndim != 4 or images.shape[1] != args.img_channels:
            raise ValueError(f"Expected BCHW images with {args.img_channels} channels; got {tuple(images.shape)}")
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 3:
            raise ValueError(f"Expected BHW labels (or B1HW); got {tuple(labels.shape)}")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("Ground-truth label must use class IDs 0/1 (as in CrossEntropyLoss), not 0/255")
        if first_benchmark_sample is None:
            first_benchmark_sample = images[:1].clone()

        with torch.inference_mode():
            logits = net(images.to(device, non_blocking=True))
            predictions = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
        truths = labels.cpu().numpy()
        names = case_names_for_batch(batch, images.shape[0], len(rows), dataset.sample_list)
        for j, name in enumerate(names):
            pred = (predictions[j] == 1)
            truth = (truths[j] == 1)
            metrics = calculate_binary_metrics(pred, truth)
            value, status = calculate_hd95(pred, truth)
            metrics["hd95"] = value
            metrics["hd95_status"] = status
            metrics["case_name"] = name
            for key in confusion:
                confusion[key] += metrics[key]
            if not args.no_save_preds:
                out_name = safe_filename(name, len(rows)) + ".png"
                Image.fromarray(pred.astype(np.uint8) * 255, mode="L").save(pred_dir / out_name)
                metrics["prediction_file"] = out_name
            rows.append(metrics)
    elapsed = time.perf_counter() - start
    if not rows:
        raise RuntimeError("Test loader yielded no samples")

    averages = {key: float(np.mean([r[key] for r in rows])) for key in METRIC_NAMES}
    hd_values = [r["hd95"] for r in rows]
    averages["hd95"] = float(np.mean(hd_values))
    statuses = {status: sum(row["hd95_status"] == status for row in rows)
                for status in ("normal", "empty_prediction", "empty_target", "both_empty")}
    benchmark = None
    if not args.no_benchmark:
        benchmark = inference_benchmark(net, first_benchmark_sample, device,
                                        warmup=args.warmup, runs=args.benchmark_runs)

    fieldnames = ["case_name", *METRIC_NAMES, "hd95", "hd95_status", "tp", "fp", "fn", "tn"]
    if not args.no_save_preds:
        fieldnames.append("prediction_file")
    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez(output_dir / "test_scores.npz",
             dice=np.array([r["dice"] for r in rows], dtype=np.float64),
             iou=np.array([r["iou"] for r in rows], dtype=np.float64),
             hd95=np.array([r["hd95"] for r in rows], dtype=np.float64),
             case_name=np.array([r["case_name"] for r in rows], dtype=str))
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", "background(0)", "lesion(1)"])
        writer.writerow(["background(0)", confusion["tn"], confusion["fp"]])
        writer.writerow(["lesion(1)", confusion["fn"], confusion["tp"]])

    summary = {
        "weights": str(Path(args.weights).resolve()),
        "checkpoint_type": source,
        "test_root": args.test_root,
        "list_dir": args.list_dir,
        "split": args.split,
        "preprocess": args.transform,
        "model_config": {"img_channels": args.img_channels,
                         "n_classes": args.num_classes, "base_c": args.base_c},
        "device": str(device),
        "sample_count": len(rows),
        "metric_aggregation": "mean of per-image scores; identical to supplied validate()",
        "metrics_fraction": averages,
        "hd95_units": "pixels at model input resolution",
        "hd95_valid_count": len(hd_values),
        "hd95_skipped_count": 0,
        "hd95_normal_count": statuses["normal"],
        "hd95_penalized_count": statuses["empty_prediction"] + statuses["empty_target"],
        "hd95_policy": "nonempty masks: medpy HD95; one empty mask: image diagonal; both empty: 0",
        "hd95_status_counts": statuses,
        "confusion_actual_rows_predicted_cols": [[confusion["tn"], confusion["fp"]],
                                                 [confusion["fn"], confusion["tp"]]],
        "end_to_end_test_seconds": elapsed,
        "forward_benchmark": benchmark,
    }
    with (output_dir / "test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print("\n" + "=" * 64)
    print(f"TEST RESULTS ({len(rows)} images, mean of per-image metrics)")
    print("=" * 64)
    for name in METRIC_NAMES:
        print(f"{name:>12s}: {averages[name] * 100:.2f}%")
    print(f"{'HD95':>12s}: {averages['hd95']:.4f} px ({len(hd_values)}/{len(rows)} included)")
    print(f"HD95 cases: normal={statuses['normal']}, empty_prediction={statuses['empty_prediction']}, "
          f"empty_target={statuses['empty_target']}, both_empty={statuses['both_empty']}")
    print(f"{'Duration':>12s}: {elapsed:.2f} s (includes data / postprocessing)")
    if benchmark is not None:
        print(f"{'Inference':>12s}: {benchmark['mean_ms_per_image']:.3f} ± "
              f"{benchmark['std_ms_per_image']:.3f} ms/image (batch=1)")
    print(f"Results saved to: {output_dir.resolve()}")
    print("HD95 policy: nonempty=MedPy; one empty mask=image diagonal; both empty=0.")
    return summary


if __name__ == "__main__":
    run_test(parse_args())
