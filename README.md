# 🩹 WoundTracker

WoundTracker is a project designed to track, manage, and predict the healing process of wounds using a data-driven approach. It utilizes machine learning to train a healing prediction model and stores detailed wound data for analysis and visualization.

---

## 🌟 Features
- **Wound Data Management:** Organize and store detailed information for multiple wounds.  
- **Healing Model Training:** Train a machine learning model (`models/healing`) to predict wound healing progression.  
- **Data Preparation:** Scripts for cleaning and preparing raw data into a usable dataset.  
- **Application Interface:** A main application file (`app.py`) for interacting with the tracker and model.  
- **Utility Functions:** Reusable scripts for common tasks.  

---

## 📁 Project Structure
woundtracker/
├── Deepskin/ # Core application/skin-related files (potential sub-module)
├── pycache/ # Python cache (ignored)
├── models/
│ └── healing/ # Trained model files for healing prediction
├── wound_data/
│ └── wound_001/ # Directory for specific wound data, images, etc.
├── .gitignore # Specifies files/folders to be ignored by Git
├── README.md # This file
├── app.py # Main application file
├── exclude.txt # File to list exclusions/ignored items
├── healing_data.csv # Raw or compiled dataset for healing
├── prepare_dataset.py # Script to process data and create the final dataset
├── requirements.txt # List of Python dependencies
├── train_healing_model.py # Script to train and save the healing prediction model
└── utils.py # General utility functions

yaml
Copy code

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ installed on your system

### Installation
1. **Clone the repository**
```bash
git clone [YOUR_REPO_URL]
cd woundtracker
Install dependencies

bash
Copy code
pip install -r requirements.txt
Usage
Prepare Data

bash
Copy code
python prepare_dataset.py
Train the Model

bash
Copy code
python train_healing_model.py
Run the Application

bash
Copy code
python app.py
🛠️ Development
Dependencies: All required libraries are in requirements.txt. Update it whenever new packages are added:

bash
Copy code
pip freeze > requirements.txt
Data Directories:

wound_data/ – Store individual wound progress files and images

models/ – Store trained machine learning models

🤝 Contributors
Diksha Agrawal (SilentAbstractDebugger)

Gowri DV (claimiris)

Aastha Sharma (Aastha0107)
