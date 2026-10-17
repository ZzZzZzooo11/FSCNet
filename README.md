# FSC-Net

This is the code repository for **"FSC-Net: A Spatial-Frequency Synergistic Network with Edge Guidance for Medical Image Segmentation"**.

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
- The manuscript reports a 7:2:1 training/validation/test split and images resized to **256 × 256**.
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

Each split file should contain one image ID per line, without the extension. Populate the split files for the data partition you intend to run; creating a new random partition may not reproduce the manuscript's reported results.

### LIDC-IDRI

- Download LIDC-IDRI from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/).
- The manuscript reports a 7:2:1 split performed **at the patient level** to avoid overlap of patients between training, validation, and test subsets.
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

We gratefully acknowledge the researchers and developers of the comparison methods used in the FSC-Net study. For convenience, we list the publicly available project/code links for all **16** comparison methods below, grouped by architectural family. The bracketed reference numbers match those in our manuscript.

**Important:** These links point to original projects or publicly available implementations for reference and attribution; they do **not**, by themselves, establish which exact repository commits, checkpoints, adaptations, or training configurations produced the results in our comparison tables. Readers should consult the individual projects for their usage instructions and licenses.

### CNN-based methods

| Method | Reference | Public code / project | Source note |
|---|---:|---|---|
| UNet | [7] | [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) | Widely used PyTorch implementation; not the original authors' release |
| DeepLabV3+ | [8] | [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) | Public PyTorch implementation |
| UNet++ | [9] | [MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus) | Original authors' implementation |
| PSPNet | [34] | [hszhao/PSPNet](https://github.com/hszhao/PSPNet) | Original authors' project; see its linked PyTorch implementation |
| ResUNet++ | [24] | [DebeshJha/ResUNetplusplus](https://github.com/DebeshJha/ResUNetplusplus) | Original author's project |
| nnU-Net | [35] | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | Official framework; note that v1 and v2 configurations differ |
| DCSAU-Net | [10] | [xq141839/DCSAU-Net](https://github.com/xq141839/DCSAU-Net) | Authors' project |
| AMSUnet | [26] | [llluochen/AMSUnet](https://github.com/llluochen/AMSUnet) | GitHub project identified in the publication |
| MSDANet | [11] | [1999luan/MSDANet](https://github.com/1999luan/MSDANet) | Public implementation; author/commit provenance should be checked |
| PDF-UNet | [36] | [ahmedeqbal/PDF-UNet](https://github.com/ahmedeqbal/PDF-UNet) | Original implementation; publication proposes a semi-supervised framework |

### Transformer-based and hybrid methods

| Method | Reference | Public code / project | Source note |
|---|---:|---|---|
| BRAU-Net++ | [25] | [Caipengzhou/BRAU-Netplusplus](https://github.com/Caipengzhou/BRAU-Netplusplus) | Project linked to the published method |
| HCT-Unet | [18] | [Author-released Zenodo code archive](https://doi.org/10.5281/zenodo.11070837) | **No matching GitHub repository verified**; original paper identifies a Zenodo archive |
| CSWin-UNet | [12] | [eatbeanss/CSWin-UNet](https://github.com/eatbeanss/CSWin-UNet) | GitHub code link provided by the paper |

### Frequency-domain and state-space methods

| Method | Reference | Public code / project | Source note |
|---|---:|---|---|
| SkinMamba | [29] | [zs1314/SkinMamba](https://github.com/zs1314/SkinMamba) | Authors' project |
| FreqConvMamba | [23] | [ccode-Rookie/FreqConvMamba](https://github.com/ccode-Rookie/FreqConvMamba) | Authors' project |

### Prompt-based foundation model

| Method | Reference | Public code / project | Source note |
|---|---:|---|---|
| SAM 2 | [37] | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Official Meta repository; results depend on the checkpoint and prompting protocol |

## 6. Acknowledgements

We sincerely thank all authors and maintainers of the above open-source projects for making their research and code publicly available. Their valuable contributions have advanced the medical image segmentation field and provided important reference architectures for our comparative study. We acknowledge the original papers and respective repositories, and encourage readers to cite the relevant works and respect their software licenses.

We also thank [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) and the [International Skin Imaging Collaboration (ISIC)](https://www.isic-archive.com/) for providing the publicly available datasets used in this study.

We particularly appreciate the authors of [DCSAU-Net](https://github.com/xq141839/DCSAU-Net), whose architecture serves as the baseline for FSC-Net's design and ablation analysis.
