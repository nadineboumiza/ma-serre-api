from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import json
import datetime
import base64
import requests as req

app = Flask(__name__)
CORS(app)

# ── Charger les modèles ───────────────────────────────────────
print("📦 Chargement des modèles ML...")

rf_model   = joblib.load('models/rf_model.joblib')
lstm_mean  = np.load('models/lstm_mean.npy')
lstm_std   = np.load('models/lstm_std.npy')
temp_coef  = np.load('models/temp_coef.npy')
data_stats = np.load('models/data_stats.npy')
conseil_model  = joblib.load('models/conseil_model.joblib')
conseil_enc    = joblib.load('models/conseil_encoder.joblib')

with open('models/conseils_dict.json', 'r', encoding='utf-8') as f:
    conseils_dict = json.load(f)

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
            '/predict/conseil  → Conseil du jour ',
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
# ROUTE 2 — Prévision légère sans TensorFlow
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

        predictions = []
        now = datetime.datetime.now()

        for i in range(1, 7):
            temp_drift  = temp_coef[0] * i * 0.1
            temp_pred   = round(temperature + temp_drift + np.random.normal(0, 0.3), 1)
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
# ROUTE 3 — Diagnostic plante (OpenRouter + Gemini)
# ══════════════════════════════════════════════════════════════
@app.route('/predict/plant', methods=['POST'])
def predict_plant():
    try:
        body       = request.get_json()
        image_b64  = body.get('image', '')
        media_type = body.get('media_type', 'image/jpeg')

        if not image_b64:
            return jsonify({'status': 'error', 'message': 'Image manquante'}), 400

        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        try:
            base64.b64decode(image_b64)
        except Exception:
            return jsonify({'status': 'error', 'message': 'Base64 invalide'}), 400

        prompt = """Tu es un expert en agronomie et maladies des plantes de serre tunisiennes.
Analyse cette photo et fournis un diagnostic en JSON avec cette structure exacte :
{
  "etat": "saine" ou "malade" ou "stress",
  "maladie": "nom ou Aucune",
  "confiance": 85,
  "symptomes": ["symptome 1", "symptome 2"],
  "causes": ["cause 1", "cause 2"],
  "traitement": ["etape 1", "etape 2"],
  "prevention": ["conseil 1", "conseil 2"],
  "urgence": "faible" ou "moyenne" ou "elevee",
  "conseil": "message court"
}
Reponds UNIQUEMENT avec le JSON valide, sans texte avant ou apres."""

        response = req.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openrouter/free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            },
            timeout=30
        )

        data = response.json()
        text = data['choices'][0]['message']['content']

        clean = text.strip()
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
            'message': f'Réponse non parseable: {str(e)}',
            'raw':     text if 'text' in locals() else ''
        }), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
# ══════════════════════════════════════════════════════════════
# ROUTE 4 — Conseil du jour (ML + Règles expertes)
# ══════════════════════════════════════════════════════════════
@app.route('/predict/conseil', methods=['POST'])
def predict_conseil():
    try:
        body = request.get_json()

        farmer_name = body.get('farmerName',  'Agriculteur')
        serre_name  = body.get('serreName',   'Ma Serre')
        temperature = float(body.get('temperature', 20))
        humidity    = float(body.get('humidity',    60))
        co2         = float(body.get('co2',         800))
        sol         = float(body.get('sol',         50))
        lumiere     = float(body.get('lumiere',     20000))
        risk        = body.get('risk',        'bon')
        disease     = body.get('disease',     'Aucune')
        temp_max    = float(body.get('tempMax', temperature + 2))

        # ── 1. Prédiction ML ─────────────────────
        X            = np.array([[temperature, humidity, co2, lumiere, sol]])
        pred_encoded = conseil_model.predict(X)[0]
        proba        = conseil_model.predict_proba(X)[0]
        label        = conseil_enc.inverse_transform([pred_encoded])[0]
        confiance    = round(float(np.max(proba)) * 100)

        # ── 2. Récupérer le conseil ───────────────
        conseil_info = conseils_dict.get(label, conseils_dict['conditions_normales'])

        # ── 3. Personnaliser le texte ─────────────
        heure        = datetime.datetime.now().strftime('%H:%M')
        conseil_text = (
            f"Bonjour {farmer_name} ! Il est {heure} et voici "
            f"votre analyse de {serre_name}.\n\n"
            f"{conseil_info['emoji']} {conseil_info['titre']}\n\n"
            f"📊 Conditions actuelles :\n"
            f"• Température : {temperature}°C (max prévue : {temp_max}°C)\n"
            f"• Humidité : {humidity}%  •  Sol : {sol}%  •  CO₂ : {co2} ppm\n"
        )

        if disease != 'Aucune':
            conseil_text += f"• ⚠️ Risque détecté : {disease}\n"

        conseil_text += f"\n✅ Actions recommandées aujourd'hui :\n"
        for i, action in enumerate(conseil_info['actions'], 1):
            conseil_text += f"{i}. {action}\n"

        conseil_text += (
            f"\n🤖 Diagnostic IA : {label.replace('_', ' ').title()} "
            f"(confiance {confiance}%)"
        )

        return jsonify({
            'status':    'ok',
            'conseil':   conseil_text,
            'label':     label,
            'urgence':   conseil_info['urgence'],
            'confiance': confiance,
            'emoji':     conseil_info['emoji'],
            'actions':   conseil_info['actions'],
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500    

# ── Lancement ─────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)