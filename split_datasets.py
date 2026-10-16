import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_ids(directory):
    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")
    ids = [p.stem for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if not ids:
        raise ValueError(f"No image files found in {directory}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs found in {directory}")
    return set(ids)


def check_masks(images_dir, masks_dir, mask_suffix):
    image_ids = read_ids(images_dir)
    mask_ids = read_ids(masks_dir)
    if mask_suffix:
        if any(not name.endswith(mask_suffix) for name in mask_ids):
            raise ValueError(f"Mask filenames must end with '{mask_suffix}' before the extension")
        mask_ids = {name[:-len(mask_suffix)] for name in mask_ids}
    if image_ids != mask_ids:
        missing = sorted(image_ids - mask_ids)
        extra = sorted(mask_ids - image_ids)
        raise ValueError(f"Image/mask IDs do not match. Missing masks: {missing[:10]}; extra masks: {extra[:10]}")
    return image_ids


def split_items(items):
    items = list(items)
    if len(items) < 3:
        raise ValueError("At least three images or patients are required")
    random.shuffle(items)
    n_train = min(len(items) - 2, max(1, round(len(items) * 0.7)))
    n_val = min(len(items) - n_train - 1, max(1, round(len(items) * 0.2)))
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def split_lidc(image_ids, mapping_csv):
    if not mapping_csv.is_file():
        raise ValueError(f"Mapping file not found: {mapping_csv}")
    patients = defaultdict(list)
    mapping = {}
    with mapping_csv.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"sample_id", "patient_id"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain sample_id and patient_id columns")
        for row in reader:
            sample_id = row["sample_id"].strip()
            patient_id = row["patient_id"].strip()
            if not sample_id or not patient_id or sample_id in mapping:
                raise ValueError(f"Invalid or duplicated mapping row: {row}")
            mapping[sample_id] = patient_id
    if image_ids != set(mapping):
        raise ValueError("Image IDs and mapping CSV sample IDs do not match")
    for sample_id, patient_id in mapping.items():
        patients[patient_id].append(sample_id)
    patient_splits = split_items(patients)
    splits = {
        name: [sample_id for patient_id in patient_ids for sample_id in patients[patient_id]]
        for name, patient_ids in patient_splits.items()
    }
    check_splits(patient_splits, set(patients))
    return splits


def check_splits(splits, original_ids):
    groups = [splits[name] for name in ("train", "val", "test")]
    sets = [set(group) for group in groups]
    if any(len(group) != len(group_set) for group, group_set in zip(groups, sets)):
        raise ValueError("Duplicate samples found within a subset")
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Duplicate samples found across subsets")
    if set.union(*sets) != original_ids:
        raise ValueError("Some samples are missing or unexpected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["isic", "lidc"])
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-suffix", default="")
    parser.add_argument("--mapping-csv", type=Path)
    args = parser.parse_args()

    image_ids = check_masks(args.images_dir, args.masks_dir, args.mask_suffix)
    if args.dataset == "lidc":
        if args.mapping_csv is None:
            parser.error("--mapping-csv is required for lidc")
        splits = split_lidc(image_ids, args.mapping_csv)
    else:
        splits = split_items(image_ids)
    check_splits(splits, image_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        (args.output_dir / f"{name}.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in splits[name]), encoding="utf-8"
        )
        print(f"{name}: {len(splits[name])}")


if __name__ == "__main__":
    main()
