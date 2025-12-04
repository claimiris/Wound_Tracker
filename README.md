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

