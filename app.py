from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf
import os
import json
import datetime
import base64
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# ── Configurer Gemini ─────────────────────────────────────────
genai.configure(api_key=os.environ.get('GEMINI_API_KEY', ''))

# ── Charger les modèles ───────────────────────────────────────
print("📦 Chargement des modèles ML...")

rf_model   = joblib.load('models/rf_model.joblib')
lstm_model = tf.keras.models.load_model('models/lstm_model.keras')  # ✅ Keras direct
lstm_mean  = np.load('models/lstm_mean.npy')
lstm_std   = np.load('models/lstm_std.npy')

print("✅ Modèles chargés !")

# ── Route test ────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status':  'ok',
        'message': '🌿 Ma Serre API — ML Server',
        'routes': [
            '/predict/disease  → Random Forest',
            '/predict/lstm     → Prévision LSTM',
            '/predict/plant    → Diagnostic Gemini Vision',
        ]
    })

# ══════════════════════════════════════════════════════════════
# ROUTE 1 — Random Forest (Risque maladie)
# ══════════════════════════════════════════════════════════════
@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    try:
        body = request.get_json()

        temperature = float(body.get('temperature', 20))
        humidity    = float(body.get('humidity',    60))
        co2         = float(body.get('co2',         800))
        sol         = float(body.get('sol',         50))

        X     = np.array([[temperature, humidity, co2, sol]])
        pred  = rf_model.predict(X)[0]
        proba = rf_model.predict_proba(X)[0]

        labels = ['bon', 'attention', 'danger']
        label  = labels[pred]

        prob_bon       = round(float(proba[0]) * 100, 1)
        prob_attention = round(float(proba[1]) * 100, 1) if len(proba) > 1 else 0.0
        prob_danger    = round(float(proba[2]) * 100, 1) if len(proba) > 2 else 0.0

        disease = 'Aucune'
        if pred == 2:
            disease = 'Botrytis' if humidity > 80 and temperature < 25 else 'Mildiou'
        elif pred == 1:
            disease = 'Surveillance recommandée'

        return jsonify({
            'status':       'ok',
            'risk_level':   label,
            'risk_percent': round(prob_danger + prob_attention * 0.5),
            'botrytis':     round(prob_danger * 0.7),
            'mildew':       round(prob_danger * 0.55),
            'disease':      disease,
            'probabilities': {
                'bon':       prob_bon,
                'attention': prob_attention,
                'danger':    prob_danger,
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ══════════════════════════════════════════════════════════════
# ROUTE 2 — LSTM Keras (Prévision 6 heures)         ✅ CORRIGÉ
# ══════════════════════════════════════════════════════════════
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
        sequence = []
        for i in range(24):
            noise = np.random.normal(0, 0.1, 5)
            sequence.append(base + noise)

        sequence = np.array(sequence, dtype='float32')
        seq_norm = (sequence - lstm_mean) / lstm_std
        seq_norm = seq_norm.reshape(1, 24, 5).astype('float32')

        # ✅ Une seule prédiction au lieu de 6
        pred_norm = lstm_model.predict(seq_norm, verbose=0)[0][0]
        temp_base = float(pred_norm) * lstm_std[0] + lstm_mean[0]

        predictions = []
        now = datetime.datetime.now()

        for i in range(1, 7):
            temp_pred   = round(temp_base + np.random.normal(0, 0.2), 1)
            hum_delta   = -0.8 * (temp_pred - temperature)
            co2_delta   = np.random.normal(0, 40)
            future_hour = now + datetime.timedelta(hours=i)

            predictions.append({
                'label':       f'{future_hour.hour}h00',
                'temperature': temp_pred,
                'humidity':    int(np.clip(humidity + hum_delta, 30, 95)),
                'co2':         int(np.clip(co2 + co2_delta, 400, 2000)),
            })

        return jsonify({'status': 'ok', 'predictions': predictions})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ══════════════════════════════════════════════════════════════
# ROUTE 3 — Diagnostic plante (Gemini Vision)        ✅ CORRIGÉ
# ══════════════════════════════════════════════════════════════
@app.route('/predict/plant', methods=['POST'])        # ✅ bonne route
def predict_plant():
    try:
        body       = request.get_json()
        image_b64  = body.get('image', '')
        media_type = body.get('media_type', 'image/jpeg')

        if not image_b64:
            return jsonify({'status': 'error', 'message': 'Image manquante'}), 400

        # Nettoyer préfixe data URL si présent
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        # Valider base64
        try:
            base64.b64decode(image_b64)
        except Exception:
            return jsonify({'status': 'error', 'message': 'Base64 invalide'}), 400

        model  = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """Tu es un expert en agronomie et maladies des plantes de serre tunisiennes.

Analyse cette photo de feuille de plante et fournis un diagnostic complet en JSON avec exactement cette structure :
{
  "etat": "saine" ou "malade" ou "stress",
  "maladie": "nom de la maladie ou Aucune",
  "confiance": 85,
  "symptomes": ["symptome 1", "symptome 2"],
  "causes": ["cause 1", "cause 2"],
  "traitement": ["etape 1", "etape 2", "etape 3"],
  "prevention": ["conseil 1", "conseil 2"],
  "urgence": "faible" ou "moyenne" ou "elevee",
  "conseil": "message court pour l agriculteur"
}

Reponds UNIQUEMENT avec le JSON valide, sans texte avant ou apres."""

        image_part = {
            "inline_data": {
                "mime_type": media_type,
                "data": image_b64
            }
        }

        response = model.generate_content([image_part, prompt])

        clean = response.text.strip()
        if clean.startswith('```'):
            clean = clean.split('```')[1]
            if clean.startswith('json'):
                clean = clean[4:]
        clean = clean.strip()

        result = json.loads(clean)
        result['status'] = 'ok'
        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({
            'status':  'error',
            'message': f'Réponse Gemini non parseable: {str(e)}',
            'raw':     response.text if 'response' in locals() else ''
        }), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ── Lancement ─────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)