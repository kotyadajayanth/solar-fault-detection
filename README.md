# ☀️ Solar Panel Fault Detection — AI-Powered Full-Stack System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![React](https://img.shields.io/badge/React-JS-61DAFB?logo=react)
![Accuracy](https://img.shields.io/badge/Accuracy-83.33%25-brightgreen)

An end-to-end AI-powered web application that detects faults in solar panels using a deep learning CNN model (MobileNetV2), served via a Flask REST API, and visualized through a React JS dashboard.

---

## 🚀 Live Demo

Upload any solar panel image → Instant fault detection with confidence score!

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | MobileNetV2 (Transfer Learning) |
| Framework | TensorFlow 2.21 / Keras |
| Input Size | 224 × 224 px |
| Classes | 6 |
| Validation Accuracy | **83.33%** |
| Training Images | 711 |

### 🔍 Fault Classes Detected

| Class | Description | Status |
|-------|-------------|--------|
| ✅ Clean | Panel working normally | Normal |
| 🌫️ Dusty | Dust on surface | Warning |
| 🐦 Bird-drop | Bird droppings | Warning |
| ⚡ Electrical-damage | Electrical fault | Danger |
| 💥 Physical-Damage | Cracks or breaks | Danger |
| ❄️ Snow-Covered | Snow on panel | Warning |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI / ML | Python, TensorFlow, Keras, MobileNetV2, CNN |
| Backend | Flask, REST API, flask-cors |
| Frontend | React JS, HTML5, CSS3 |
| Tools | Git, GitHub, Scikit-learn, NumPy, Pillow |

---

## 📁 Project Structure
solar-fault-detection/
├── dataset/                  # Solar panel images (6 classes)
│   ├── Clean/
│   ├── Dusty/
│   ├── Bird-drop/
│   ├── Electrical-damage/
│   ├── Physical-Damage/
│   └── Snow-Covered/
├── model/
│   ├── train.py              # CNN training script (MobileNetV2)
│   ├── solar_model.keras     # Trained model (saved after training)
│   └── training_results.png # Accuracy/loss graph
├── backend/
│   ├── app.py                # Flask REST API
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js            # Main React component
│   │   └── App.css           # Styling
│   └── package.json
└── README.md

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/kotyadajayanth/solar-fault-detection.git
cd solar-fault-detection
```

### 2️⃣ Install Python Dependencies
```bash
pip install tensorflow keras flask flask-cors numpy pillow scikit-learn matplotlib
```

### 3️⃣ Train the Model (Optional — skip if model exists)
```bash
python model/train.py
```

### 4️⃣ Start Flask Backend
```bash
python backend/app.py
```
API runs on → **http://127.0.0.1:8000**

### 5️⃣ Start React Frontend
```bash
cd frontend
npm install
npm start
```
App runs on → **http://localhost:3000**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status check |
| POST | `/predict` | Upload image → get fault prediction |

### Example Response
```json
{
  "predicted_class": "Dusty",
  "confidence": 91.45,
  "status": "warning",
  "message": "Dust detected. Clean panel for better efficiency.",
  "all_probabilities": {
    "Bird-drop": 2.31,
    "Clean": 3.12,
    "Dusty": 91.45,
    "Electrical-damage": 1.02,
    "Physical-Damage": 0.89,
    "Snow-Covered": 1.21
  }
}
```

---

## 📊 Training Results

- Phase 1 (Frozen base): Top layers trained for feature extraction
- Phase 2 (Fine-tuning): Last 30 layers unfrozen for domain adaptation
- **Best Validation Accuracy: 83.33%**

---

## 👨‍💻 Author

**Jayanth Kotyada**
- 🔗 [LinkedIn](https://linkedin.com/in/jayanth-kotyada)
- 🐙 [GitHub](https://github.com/kotyadajayanth)
- 📧 kotyadajayanth5@gmail.com

---
