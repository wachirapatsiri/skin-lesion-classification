# 🔬 Skin Lesion Classification with Explainable AI (HAM10000)

## 📌 Project Overview
An end-to-end computer vision pipeline and Gradio web application for classifying skin lesions from dermoscopic images using the HAM10000 dataset, achieving high diagnostic sensitivity and interpretability.

## 🎯 Business Problem
Early and accurate detection of malignant skin lesions (like melanoma) is critical for patient survival. However, dermoscopic diagnosis is challenging and highly dependent on dermatologist expertise. Furthermore, medical datasets often suffer from severe class imbalance, making standard AI models prone to misdiagnosis. 

## 🛠️ Solution & Action (Tech Stack: PyTorch, EfficientNet-B0, Grad-CAM, Gradio)
*   **Data Pipeline & Augmentation:** Processed the HAM10000 dataset (**10K** images) and implemented a robust data pipeline with PyTorch transforms (Random Resized Crop, Color Jitter, Random Erasing) to combat overfitting.
*   **Advanced Modeling & Imbalance Handling:** Trained an EfficientNet-B0 model. To address the severe class imbalance (58.3:1 ratio), I experimented with multiple techniques including Class-Weighted CE, Focal Loss + Balanced Samplers, and Post-hoc Logit Adjustment.
*   **Multi-Modal Fusion:** Developed an advanced architecture integrating both image data and tabular patient metadata (age, sex, localization) to boost model performance.
*   **Explainable AI (XAI):** Integrated Grad-CAM to visually explain model predictions, highlighting the specific regions of the lesion the model focused on.
*   **Deployment:** Built an interactive Gradio interface for clinical screening demonstration.

## 📈 Key Results
*   The baseline model achieved a **Macro-F1 score of 0.6486**.
*   The Multi-Modal Fusion approach (Image + Metadata) significantly improved performance, achieving a **Macro-F1 of 0.7001** and an **Accuracy of 82.76%**.
*   Successfully established a high-sensitivity clinical screening threshold (95% Sensitivity) to minimize missed malignant cases.

## 💻 Live Demo & UI
