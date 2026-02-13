# Pneumonia-Classification-with-ResNet18

# 🩺 Pneumonia Classification with ResNet18

Deep learning–based pneumonia detection from chest X-ray images using a fine-tuned ResNet18 architecture.

---

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) model based on **ResNet18** to classify chest X-ray images into two categories:

- **Pneumonia**
- **Normal**

The goal is to build a reproducible deep learning pipeline for medical image classification using transfer learning and evaluate performance using standard classification metrics.

---

## 🧠 Methodology

### 🔹 Model Architecture
- Pretrained **ResNet18**
- Replaced final fully connected (FC) layer
- Transfer learning approach

### 🔹 Training Configuration
- Loss Function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning Rate: `1e-4`
- Batch Size: `32`
- Image Size: `224x224`
- Train / Validation Split: `80 / 20`
- Data Augmentation:
  - RandomHorizontalFlip
  - RandomRotation
  - Resize + Normalize

---

## 📊 Model Performance

| Metric     | Score |
|------------|--------|
| Accuracy   | XX% |
| Precision  | XX |
| Recall     | XX |
| F1-score   | XX |
| AUC        | XX |

> Replace `XX` with your actual results after training.

You may also include:
- Confusion Matrix
- ROC Curve
- Training / Validation Loss curves

---

## 📂 Project Structure

```
Pneumonia-Classification-with-ResNet18/
│
├── notebooks/
│   └── Pneumonia_Classification_ResNet_Tutorial.ipynb
│
├── src/
│   ├── train.py
│   ├── dataset.py
│   └── utils.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Pneumonia-Classification-with-ResNet18.git
cd Pneumonia-Classification-with-ResNet18
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Dataset

Download the Chest X-ray dataset and organize it as:

```
data/
 ├── train/
 │   ├── NORMAL/
 │   └── PNEUMONIA/
 ├── val/
 │   ├── NORMAL/
 │   └── PNEUMONIA/
```

---

### 4️⃣ Run Training

If using the notebook:

```
Open notebooks/Pneumonia_Classification_ResNet_Tutorial.ipynb
```

If using Python script:

```bash
python src/train.py
```

---

## 🏥 Dataset

- Public chest X-ray dataset for pneumonia detection
- Binary classification: Pneumonia vs Normal

(Insert dataset source link here)

---

## 🛠 Tech Stack

- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn

---

## 🎯 Key Learnings

- Applied transfer learning to medical image classification
- Built a structured deep learning training pipeline
- Evaluated model performance using multiple metrics
- Organized project for reproducibility and scalability

---

## 🔮 Future Improvements

- Hyperparameter tuning
- Grad-CAM for model interpretability
- Handling class imbalance
- Cross-validation
- Model deployment (Streamlit / FastAPI)

---

If you found this project helpful, feel free to ⭐ the repository!
