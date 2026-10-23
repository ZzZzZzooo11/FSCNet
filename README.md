# FSC-Net

This is the code repository for **"FSC-Net: Gated spatial–frequency fusion with edge-guided refinement for medical image segmentation"**.

FSC-Net combines **Discrete Wavelet Transform (DWT)** downsampling, a **Spatial-Frequency Feature Module (SFFM)**, and an **Edge-Guided Attention Block (EGAB)** for pulmonary nodule and skin lesion segmentation.

## 0. Main Environments

The following core versions are taken from the project Conda environment:

- Python 3.9.25
- PyTorch 2.5.1
- CUDA Toolkit 11.8

Other main Python dependencies include `numpy`, `scipy`, `Pillow`, `h5py`, `medpy`, `matplotlib`, `scikit-learn`, `ptflops`, `tensorboardX`, and `tqdm`.

## 1. Prepare the Datasets

### ISIC 2018

- Download the ISIC 2018 skin lesion segmentation dataset from the [ISIC Archive](https://challenge.isic-archive.com/data/).
- 7:2:1 training/validation/test split and images resized to **256 × 256**.
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
- 7:2:1 split performed **at the patient level** to avoid overlap of patients between training, validation, and test subsets.
- Prepare the 2D image/mask pairs and matching patient-disjoint split lists before training or testing.

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

**Note:** The commands above are example entry points, not a claim that the current public checkout has been independently validated end-to-end. Check local data paths, dependencies, preprocessing, and checkpoint configuration before use. For a scientifically comparable evaluation, use the same data partition and testing protocol across methods.

## 4. Obtain the Outputs

- Training logs and checkpoints: `logs/FSC-Net/FSC-Net/`
- Test metrics and predicted masks: `logs/FSC-Net/FSC-Net/test_results/`

## 5. Comparison Methods and Open-Source References

We gratefully acknowledge the researchers and developers of the comparison methods used in the FSC-Net study. For convenience, we list the publicly available project/code links for all **16** comparison methods below, grouped by architectural family.

### CNN-based methods

| Method | Public code / project |
|---|---|
| UNet | [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) |
| DeepLabV3+ | [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) |
| UNet++ | [MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus) |
| PSPNet | [hszhao/PSPNet](https://github.com/hszhao/PSPNet) |
| ResUNet++ | [DebeshJha/ResUNetplusplus](https://github.com/DebeshJha/ResUNetplusplus) |
| nnU-Net | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| DCSAU-Net | [xq141839/DCSAU-Net](https://github.com/xq141839/DCSAU-Net) |
| AMSUnet | [llluochen/AMSUnet](https://github.com/llluochen/AMSUnet) |
| MSDANet | [1999luan/MSDANet](https://github.com/1999luan/MSDANet) |
| PDF-UNet | [ahmedeqbal/PDF-UNet](https://github.com/ahmedeqbal/PDF-UNet) |

### Transformer-based and hybrid methods

| Method | Public code / project |
|---|---|
| BRAU-Net++ | [Caipengzhou/BRAU-Netplusplus](https://github.com/Caipengzhou/BRAU-Netplusplus) |
| HCT-Unet | [Author-released Zenodo code archive](https://doi.org/10.5281/zenodo.11070837) |
| CSWin-UNet | [eatbeanss/CSWin-UNet](https://github.com/eatbeanss/CSWin-UNet) |

### Frequency-domain and state-space methods

| Method | Public code / project |
|---|---|
| SkinMamba | [zs1314/SkinMamba](https://github.com/zs1314/SkinMamba) |
| FreqConvMamba | [ccode-Rookie/FreqConvMamba](https://github.com/ccode-Rookie/FreqConvMamba) |

### Prompt-based foundation model

| Method | Public code / project |
|---|---|
| SAM 2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |

## 6. Acknowledgements

We sincerely thank all authors and maintainers of the above open-source projects for making their research and code publicly available. Their valuable contributions have advanced the medical image segmentation field and provided important reference architectures for our comparative study. We acknowledge the original papers and respective repositories, and encourage readers to cite the relevant works and respect their software licenses.

We also thank [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) and the [International Skin Imaging Collaboration (ISIC)](https://www.isic-archive.com/) for providing the publicly available datasets used in this study.

We particularly appreciate the authors of [DCSAU-Net](https://github.com/xq141839/DCSAU-Net), whose architecture serves as the baseline for FSC-Net's design and ablation analysis.
