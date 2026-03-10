from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

# ── Charger les modèles ───────────────────────────────
print("📦 Chargement des modèles ML...")

rf_model   = joblib.load('models/rf_model.joblib')
lstm_model = tf.keras.models.load_model('models/lstm_model.keras')
lstm_mean  = np.load('models/lstm_mean.npy')
lstm_std   = np.load('models/lstm_std.npy')

print("✅ Modèles chargés !")

# ── Route test ────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status':  'ok',
        'message': '🌿 Ma Serre API — ML Server',
        'routes': [
            '/predict/disease  → Random Forest',
            '/predict/lstm     → Prévision LSTM',
        ]
    })

# ═══════════════════════════════════════════════════
# ROUTE 1 — Random Forest (Risque maladie)
# ═══════════════════════════════════════════════════
@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    try:
        body = request.get_json()

        temperature = float(body.get('temperature', 20))
        humidity    = float(body.get('humidity',    60))
        co2         = float(body.get('co2',         800))
        sol         = float(body.get('sol',         50))

        # Prédiction
        X       = np.array([[temperature, humidity, co2, sol]])
        pred    = rf_model.predict(X)[0]
        proba   = rf_model.predict_proba(X)[0]

        labels  = ['bon', 'attention', 'danger']
        label   = labels[pred]

        # Probabilités
        prob_bon       = round(float(proba[0]) * 100, 1)
        prob_attention = round(float(proba[1]) * 100, 1) \
            if len(proba) > 1 else 0.0
        prob_danger    = round(float(proba[2]) * 100, 1) \
            if len(proba) > 2 else 0.0

        # Maladie probable
        disease = 'Aucune'
        if pred == 2:
            disease = 'Botrytis' \
                if humidity > 80 and temperature < 25 \
                else 'Mildiou'
        elif pred == 1:
            disease = 'Surveillance recommandée'

        return jsonify({
            'status':          'ok',
            'risk_level':      label,
            'risk_percent':    round(prob_danger + prob_attention * 0.5),
            'botrytis':        round(prob_danger * 0.7),
            'mildew':          round(prob_danger * 0.55),
            'disease':         disease,
            'probabilities': {
                'bon':       prob_bon,
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

        # Créer séquence de 24 pas simulée
        base = np.array([temperature, humidity,
                         co2, lumiere, sol])
        sequence = []
        for i in range(24):
            noise = np.random.normal(0, 0.1, 5)
            sequence.append(base + noise)
        sequence = np.array(sequence, dtype='float32')

        # Normaliser
        seq_norm = (sequence - lstm_mean) / lstm_std
        seq_norm = seq_norm.reshape(1, 24, 5)

        # Prédictions 6 heures
        predictions = []
        now         = __import__('datetime').datetime.now()

        for i in range(1, 7):
            # Décaler la séquence
            pred_norm  = lstm_model.predict(
                seq_norm, verbose=0)[0][0]

            # Dénormaliser
            temp_pred  = pred_norm * lstm_std[0] + lstm_mean[0]

            # Humidité et CO₂ inversement corrélés
            hum_delta  = -0.8 * (temp_pred - temperature)
            co2_delta  = np.random.normal(0, 40)

            future_hour = now + \
                __import__('datetime').timedelta(hours=i)

            predictions.append({
                'label':       f'{future_hour.hour}h00',
                'temperature': round(float(temp_pred), 1),
                'humidity':    int(np.clip(
                    humidity + hum_delta, 30, 95)),
                'co2':         int(np.clip(
                    co2 + co2_delta, 400, 2000)),
            })

            # Mettre à jour séquence
            new_point  = np.array([
                temp_pred, humidity + hum_delta,
                co2 + co2_delta, lumiere, sol
            ], dtype='float32')
            new_norm   = (new_point - lstm_mean) / lstm_std
            seq_norm   = np.roll(seq_norm, -1, axis=1)
            seq_norm[0, -1, :] = new_norm

        return jsonify({
            'status':      'ok',
            'predictions': predictions,
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ── Lancement ─────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)