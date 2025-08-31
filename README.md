# Sign Language Recognition Project

## 📖 Overview
This project implements a deep learning pipeline for recognizing sign language gestures.  
It includes:
- Data preprocessing and augmentation
- Model training and evaluation
- Inference on new images or videos
- Visualization of results

The project is designed to be **modular, reproducible, and extensible** for further experimentation.

## Models

We provide implementations of the following models:
- Custom CNN
- MobileNetV2
- ResNet50
- AlexNet

Each model is implemented inside the `models/` folder, and their respective training functions are in `train.py`.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Purgty/ASL-Recognition-Experimentation.git
cd sign-language-recognition
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# Activate:
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
This project supports **Kaggle datasets**.  
Make sure you have your Kaggle API key configured in utils. To download:
```bash
python utils/dataset_loader.py
```

---

## ▶️ Running the Project


Each model is implemented inside the `models/` folder, and their respective training functions are in `trainers/train.py`.

### 1. Data Preparation
```bash
python scripts/data_preprocessing.py
```

### 2. Training
```bash
python train.py --epochs 50 --batch-size 32 --lr 0.001
```

### 3. Evaluation
```bash
python evaluate.py --model checkpoints/best_model.pth
```

### 4. Inference
To run inference on an image/video:
```bash
python inference.py --input sample.jpg --model checkpoints/best_model.pth
```

---

## 📊 Results
Here is an example of inference results across different models:

![Inference Results]("https://github.com/user-attachments/assets/46a5ea16-7d91-4766-a16f-f6c227bbb3db")

---

## 🛠️ Requirements
- Python 3.8+
- PyTorch / TensorFlow (depending on implementation)
- OpenCV
- NumPy, Pandas, Matplotlib
- Kaggle API (for dataset download)

Install via:
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure
```
sign_language_project/
│── data/                   # Datasets
│── scripts/                # Data preprocessing scripts
│── models/                 # Model definitions
│── checkpoints/            # Saved models
│── results/                # Evaluation and inference outputs
│── train.py                # Training script
│── evaluate.py             # Evaluation script
│── inference.py            # Inference script
│── requirements.txt        # Dependencies
│── README.md               # Project documentation
```

---

## ✨ Future Work
- Extend to real-time recognition via webcam
- Add more robust gesture datasets
- Improve model efficiency for deployment on edge devices

---

## 👨‍💻 Author
Your Name — [your.email@example.com]
