import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH = r"C:\Users\Admin\OneDrive\Desktop\Projects\solar-fault-detection\model\solar_model.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = {
    0: 'Bird-drop',
    1: 'Clean',
    2: 'Dusty',
    3: 'Electrical-damage',
    4: 'Physical-Damage',
    5: 'Snow-Covered'
}

CLASS_INFO = {
    'Bird-drop':          {'status': 'warning', 'message': 'Bird droppings detected. Clean the panel soon.'},
    'Clean':              {'status': 'success', 'message': 'Panel is clean and working normally.'},
    'Dusty':              {'status': 'warning', 'message': 'Dust detected. Clean panel for better efficiency.'},
    'Electrical-damage':  {'status': 'danger',  'message': 'Electrical damage detected. Immediate inspection required!'},
    'Physical-Damage':    {'status': 'danger',  'message': 'Physical damage detected. Repair needed immediately!'},
    'Snow-Covered':       {'status': 'warning', 'message': 'Snow covering detected. Remove snow for normal operation.'}
}

# ─── LOAD MODEL ───────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ─── ROUTES ───────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Solar Panel Fault Detection API',
        'status': 'running',
        'endpoints': {
            'predict': 'POST /predict'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        predicted_class = CLASS_NAMES[predicted_class_idx]
        info = CLASS_INFO[predicted_class]

        # All class probabilities
        all_probs = {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'status': info['status'],
            'message': info['message'],
            'all_probabilities': all_probs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── RUN ──────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=8000)