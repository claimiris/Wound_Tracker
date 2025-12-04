# 🩹 Wound Healing Progress Tracker  
Deep Learning Pipeline for Wound Segmentation & Healing Percentage Prediction

## 📌 Overview
This project builds a complete machine learning pipeline to automatically 
estimate how much a wound has healed over time using image-based analysis.

It combines:
- **Semantic segmentation** (DeepSkin – TensorFlow)
- **Dataset generation** (Python)
- **Regression-based healing prediction** (PyTorch, ResNet18)

The system takes a time-series of wound photographs and outputs:
- Wound segmentation mask  
- Wound area  
- Healing % relative to Day 1  
- A deep model's prediction of healing progress  

---

## 🧱 Project Structure
```
woundtracker/
│
├── Deepskin/                     # DeepSkin segmentation model (TensorFlow)
│   └── models/
│
├── app.py                        # inference app
├── predictor.py                  # Converts segmentation masks → healing_data.csv
├── train_healing_model.py        # Trains the HealingPredictor model
├── utils.py                      # Helper functions
│
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
│
└── wounds_sample/                # Example dataset (dummy images)

```
---

## 📊 Pipeline Details

### **1. Wound Segmentation – DeepSkin (TensorFlow)**
- Architecture: EfficientNet-B3 encoder + custom decoder (UNet-like)
- Output: 3-channel softmax mask
- Extracts wound region for accurate area estimation.

### **2. Dataset Preparation**
`prepare_dataset.py`:
- Loads each wound image  
- Runs DeepSkin to generate the segmentation mask  
- Computes wound area  
- Computes healing % = 1 − (area_today / area_day1)  
- Saves all rows into `healing_data.csv`  

CSV format:
wound_id,img_path,day,wound_area,healing_pct


### **3. Healing Prediction Model (PyTorch)**
- Backbone: ResNet-18 pretrained on ImageNet
- Head: 512 → 128 → 1 regression MLP
- Target: Healing percentage (0–1 range)

Loss: `MSELoss`  
Optimizer: `Adam`  

---

## 📁 Required Dataset Structure

woundsdata/
wound1/
d1.png
d2.png
d3.png
wound2/
d1.png
d2.png
d3.png
...
wound23/
Run:
python prepare_dataset.py

makefile

Produces:
healing_data.csv

Then train:
python train_healing_model.py

---

## ⚙️ Installation

pip install -r requirements.txt

---

## 🚀 Running the Full Pipeline

### **Step 1 — Prepare data**
python prepare_dataset.py


### **Step 2 — Train Healing Predictor**
python train_healing_model.py

---

## 📦 Model Weights

Weights are **not included** in this repository due to size.

Add your model weights to:
models/unet_model.pth
models/healing_predictor.pth

---

## 🧪 Expected Results
- Segmentation IoU: ~85–95% (DeepSkin benchmark)
- Healing prediction: ±5–10% MAE
- Automated healing monitoring from raw images

---
