# Breast Cancer Detection Using CNN

This project aims to **classify breast cancer histopathology images** into two categories:  
- `0` → **Non-Invasive Ductal Carcinoma (non-IDC)**  
- `1` → **Invasive Ductal Carcinoma (IDC)**  

The model is trained on a large dataset of image patches extracted from breast cancer whole-slide images.

---

## Dataset

- **Source**: [IDC Regular Breast Cancer Histopathology Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
- **Total Patches**: 277,524 (50x50 RGB PNG images)
  - 198,738 Non-IDC
  - 78,786 IDC

---

## Model Architecture

A simple yet effective **Convolutional Neural Network (CNN)** built using TensorFlow/Keras:


- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Evaluation: Accuracy, Confusion Matrix, Classification Report

---

## Results

> Trained for 10 epochs on 70% of the data.
| Metric                   | Expected Range |
| ------------------------ | -------------- |
| **Accuracy**             | 75% - 85%      |
| **Precision**            | 70% - 85%      |
| **Recall (Sensitivity)** | 70% - 85%      |
| **F1 Score**             | 72% - 85%      |
| **AUC-ROC**              | 0.80 - 0.90    |


Loss and Accuracy plots are included in `evaluate.py`.

