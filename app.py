from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import json

app = Flask(__name__)
CORS(app)

# ── Charger les modèles ───────────────────────────────
print("📦 Chargement des modèles ML...")

# Modèles de données (IoT)
rf_model    = joblib.load('models/rf_model.joblib')
lstm_model  = tf.keras.models.load_model('models/lstm_model.keras')
lstm_mean   = np.load('models/lstm_mean.npy')
lstm_std    = np.load('models/lstm_std.npy')

# Nouveau Modèle Vision (Scan feuilles)
leaf_model   = tf.keras.models.load_model('models/plant_disease_model.keras')

with open('models/plant_classes.json', 'r', encoding='utf-8') as f:
    plant_classes = json.load(f)

with open('models/disease_info.json', 'r', encoding='utf-8') as f:
    disease_info = json.load(f)

print("✅ Tous les modèles (IoT + Vision) sont chargés !")

# ── Route test ────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status':  'ok',
        'message': '🌿 Ma Serre API — ML Server (IoT & Vision)',
        'routes': [
            '/predict/disease  → Random Forest',
            '/predict/lstm     → Prévision LSTM',
            '/predict/leaf     → Scan Image (CNN)'
        ]
    })

# ═══════════════════════════════════════════════════
# ROUTE 1 — Random Forest (Risque maladie via capteurs)
# ═══════════════════════════════════════════════════
@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    try:
        body = request.get_json()
        temperature = float(body.get('temperature', 20))
        humidity    = float(body.get('humidity',    60))
        co2         = float(body.get('co2',         800))
        sol         = float(body.get('sol',         50))

        X       = np.array([[temperature, humidity, co2, sol]])
        pred    = rf_model.predict(X)[0]
        proba   = rf_model.predict_proba(X)[0]

        labels  = ['bon', 'attention', 'danger']
        label   = labels[pred]

        prob_danger    = round(float(proba[2]) * 100, 1) if len(proba) > 2 else 0.0
        prob_attention = round(float(proba[1]) * 100, 1) if len(proba) > 1 else 0.0

        disease = 'Aucune'
        if pred == 2:
            disease = 'Botrytis' if humidity > 80 and temperature < 25 else 'Mildiou'
        elif pred == 1:
            disease = 'Surveillance recommandée'

        return jsonify({
            'status':          'ok',
            'risk_level':      label,
            'risk_percent':    round(prob_danger + prob_attention * 0.5),
            'disease':         disease,
            'probabilities': {
                'bon':       round(float(proba[0]) * 100, 1),
                'attention': prob_attention,
                'danger':    prob_danger,
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ═══════════════════════════════════════════════════
# ROUTE 2 — LSTM (Prévision 6 heures)
# ═══════════════════════════════════════════════════
@app.route('/predict/lstm', methods=['POST'])
def predict_lstm():
    try:
        body    = request.get_json()
        current = body.get('current', {})
        temperature = float(current.get('temperature', 20))
        humidity    = float(current.get('humidity',    60))
        co2         = float(current.get('co2',         800))
        lumiere     = float(current.get('lumiere',     20000))
        sol         = float(current.get('sol',         50))

        base = np.array([temperature, humidity, co2, lumiere, sol])
        sequence = [base + np.random.normal(0, 0.1, 5) for _ in range(24)]
        seq_norm = (np.array(sequence, dtype='float32') - lstm_mean) / lstm_std
        seq_norm = seq_norm.reshape(1, 24, 5)

        predictions = []
        now = __import__('datetime').datetime.now()

        for i in range(1, 7):
            pred_norm  = lstm_model.predict(seq_norm, verbose=0)[0][0]
            temp_pred  = pred_norm * lstm_std[0] + lstm_mean[0]
            future_hour = now + __import__('datetime').timedelta(hours=i)

            predictions.append({
                'label': f'{future_hour.hour}h00',
                'temperature': round(float(temp_pred), 1),
                'humidity': int(np.clip(humidity - 0.8 * (temp_pred - temperature), 30, 95)),
            })
            
            new_point = np.array([temp_pred, humidity, co2, lumiere, sol], dtype='float32')
            new_norm  = (new_point - lstm_mean) / lstm_std
            seq_norm  = np.roll(seq_norm, -1, axis=1)
            seq_norm[0, -1, :] = new_norm

        return jsonify({'status': 'ok', 'predictions': predictions})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ═══════════════════════════════════════════════════
# ROUTE 3 — Scan de Feuille (CNN Vision)
# ═══════════════════════════════════════════════════
@app.route('/predict/leaf', methods=['POST'])
def predict_leaf():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'Image manquante'}), 400
        
        file = request.files['file']
        temp_path = "temp_scan.jpg"
        file.save(temp_path)

        # Prétraitement
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Inférence
        preds = leaf_model.predict(img_array)
        idx = np.argmax(preds[0])
        conf = float(np.max(preds[0]) * 100)

        class_name = plant_classes[idx]
        info = disease_info.get(class_name, {})

        if os.path.exists(temp_path): os.remove(temp_path)

        return jsonify({
            'status': 'ok',
            'prediction': {
                'plante': info.get('plante', 'Inconnue'),
                'maladie': info.get('maladie', 'Inconnue'),
                'confidence': round(conf, 1),
                'statut': info.get('statut', 'attention'),
                'traitement': info.get('traitement', []),
                'prevention': info.get('prevention', '')
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)