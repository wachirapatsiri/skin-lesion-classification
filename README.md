# Skin Lesion Classification with Multimodal Fusion

Classifying skin lesions (HAM10000) using a CNN image backbone fused with patient metadata, supported by Grad-CAM visual explanations.

**Live/Demo repo:** [github.com/wachirapatsiri/skin-lesion-classification](https://github.com/wachirapatsiri/skin-lesion-classification)

## Problem
Dermatological diagnosis requires both visual inspection of lesions and patient context (such as age, sex, and anatomical site). This project explores whether fusing image features with tabular metadata improves classification accuracy over image-only models.

## Data & Preprocessing
- **Dataset:** [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) (7 disease classes, 10,155 images).
- **Split Strategy:** Grouped by `lesion_id` to prevent data leakage (images of the same lesion never cross between train/val/test).
- **Split Ratio:** Train (6,981), Validation (1,532), Test (1,502).
- **Metadata:** Age (standardized) and One-Hot Encoded sex & localization (19 dimensions total).

## Approach
1. Trained an image-only EfficientNet-B0 baseline (Experiment A).
2. Experimented with class weighting, focal loss, balanced sampling, and logit adjustment to handle severe class imbalance (Experiments B–E).
3. Built a **Multimodal Fusion Model (Experiment F):** concatenated EfficientNet-B0 image features with a processed metadata MLP block.
4. Evaluated models using Macro-F1, Balanced Accuracy, and clinical screening thresholds tuned on the validation set.

## Results

| Model / Experiment | Accuracy | Macro-F1 | Balanced Acc | Melanoma Recall |
|---|---|---|---|---|
| A: Baseline (Image-only) | 0.809 | 0.656 | 0.622 | 0.521 |
| B: Class-Weighted CE | 0.492 | 0.439 | 0.647 | 0.569 |
| C: Focal Loss + Sampler | 0.386 | 0.417 | 0.632 | 0.653 |
| D: Logit Adjustment ($\tau=0.20$) | 0.800 | 0.642 | 0.647 | 0.539 |
| E: Logit Adjustment ($\tau=0.85$) | 0.703 | 0.527 | 0.688 | 0.587 |
| **F: Multimodal Fusion** | **0.829** | **0.702** | **0.694** | **0.545** |

- **Key Finding:** Metadata fusion improved Macro-F1 by **+0.046** over the image-only baseline, outperforming loss-engineering methods that severely traded off overall accuracy.
- **Clinical Screening (Malignant detection):** Tuning the decision threshold on the validation set to achieve 95% sensitivity yielded a test specificity of 0.628 (Malignant ROC-AUC 0.928).

## Limitations
- While metadata fusion improves overall macro-metrics, per-class recall for minority classes like melanoma remains constrained (0.545), reflecting the difficulty of visual-only feature extraction on dermatoscopic images.
- Educational project. Not intended for clinical diagnosis.

## Tech Stack
Python, PyTorch, timm (EfficientNet), scikit-learn, pandas, Grad-CAM

## 💻 Live Demo & UI
<img width="1900" height="942" alt="image" src="https://github.com/user-attachments/assets/c348995d-e96c-46f0-9e85-0ef5d0d55b03" />
