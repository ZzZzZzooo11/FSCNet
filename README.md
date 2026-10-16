# FSC-Net

This is the code repository for **"FSC-Net: A Spatial-Frequency Synergistic Network with Edge Guidance for Medical Image Segmentation"**.

FSC-Net combines **Discrete Wavelet Transform (DWT)** downsampling, a **Spatial-Frequency Feature Module (SFFM)**, and an **Edge-Guided Attention Block (EGAB)** for pulmonary nodule and skin lesion segmentation.

## 0. Main Environments

The following core versions are taken from the provided Conda environment:

- Python 3.10.21
- PyTorch 2.13.0+cu129
- torchvision 0.28.0+cu129
- CUDA Toolkit 12.9

Other main Python dependencies include `numpy`, `scipy`, `Pillow`, `h5py`, `medpy`, `matplotlib`, `scikit-learn`, `ptflops`, `tensorboardX`, and `tqdm`.

## 1. Prepare the Datasets

### ISIC 2018

- Download the ISIC 2018 skin lesion segmentation dataset from the [ISIC Archive](https://challenge.isic-archive.com/data/).
- The manuscript uses a 7:2:1 training/validation/test split and images resized to **256 × 256**.
- For the current dataset loader, organize the files as follows:

```text
data/isic2018/
├── train/
│   ├── images/    # .jpg / .png / .jpeg
│   └── masks/     # .png
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/

lists/isic2018_list/
├── train.txt
├── val.txt
└── test.txt
```

Each split file should contain one image ID per line, without the extension. 

### LIDC-IDRI

- Download LIDC-IDRI from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/).
- The 7:2:1 split is performed **at the patient level**, to prevent data leakage across subsets.

## 2. Train FSC-Net

After preparing the ISIC 2018 data and resolving any missing local code dependencies, use the training entry point:

```bash
python train.py --root_path data/isic2018/train \
                --list_dir lists/isic2018_list \
                --max_epochs 200 --batch_size 8 --img_size 256
```

## 3. Test FSC-Net

After training and obtaining a valid checkpoint:

```bash
python test.py --test_root data/isic2018/test \
               --list_dir lists/isic2018_list \
               --weights logs/FSC-Net/FSC-Net/best_model.pth \
               --transform deterministic
```

**Note:** These are intended script entry points, not verified end-to-end commands for the current public checkout. The repository's model imports and helper-module dependencies must be checked before running. The effective learning-rate setting in `trainer.py` should also be reconciled with the manuscript.

## 4. Obtain the Outputs

- Training logs and checkpoints: `logs/FSC-Net/FSC-Net/`
- Test metrics and predicted masks: `logs/FSC-Net/FSC-Net/test_results/`

## 5. Acknowledgements

We thank [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) and the [International Skin Imaging Collaboration (ISIC)](https://www.isic-archive.com/) for providing the datasets used in this study.
