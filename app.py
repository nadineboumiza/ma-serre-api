from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import json
import datetime
import base64
import requests as req
import pandas as pd

app = Flask(__name__)
CORS(app)

# ── Charger les modeles ───────────────────────────────────────
print("Chargement des modeles ML...")

rf_model   = joblib.load('models/rf_model.joblib')
lstm_mean  = np.load('models/lstm_mean.npy')
lstm_std   = np.load('models/lstm_std.npy')
temp_coef  = np.load('models/temp_coef.npy')
data_stats = np.load('models/data_stats.npy')

# Charger le vrai LSTM
try:
    from tensorflow import keras
    lstm_model = keras.models.load_model('models/lstm_model.keras')
    USE_LSTM = True
    print("LSTM charge !")
except Exception as e:
    lstm_model = None
    USE_LSTM   = False
    print(f"LSTM absent — fallback modele leger")

# Charger le modele conseil
try:
    conseil_model = joblib.load('models/conseil_model.joblib')
    conseil_enc   = joblib.load('models/conseil_encoder.joblib')
    print("Modele conseil charge !")
except FileNotFoundError:
    conseil_model = None
    conseil_enc   = None
    print("Modele conseil absent — route /predict/conseil desactivee")

with open('models/conseils_dict.json', 'r', encoding='utf-8') as f:
    conseils_dict = json.load(f)

print("Tous les modeles charges !")

# ── Route test ────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status':  'ok',
        'message': 'Ma Serre API — ML Server',
        'lstm':    'actif' if USE_LSTM else 'fallback leger',
        'routes':  [
            '/predict/disease',
            '/predict/lstm',
            '/predict/plant',
            '/predict/conseil',
            '/sync',
        ]
    })

# ═════════════════════════════════════════════════════════════
# SEUILS AGRONOMIQUES (source : seuil_et_bioagresseur.pptx)
# ═════════════════════════════════════════════════════════════
def detect_diseases(temperature, humidity, co2, sol):
    diseases = []
    risk_max = 0

    # 1. MILDIOU
    mildiou_risk = 0
    if humidity > 90 and 10 <= temperature <= 25:    mildiou_risk = 90
    elif humidity > 85 and 10 <= temperature <= 25:  mildiou_risk = 65
    elif humidity > 80 and 10 <= temperature <= 25:  mildiou_risk = 40
    elif humidity > 75 and 10 <= temperature <= 25:  mildiou_risk = 20
    if mildiou_risk > 0:
        diseases.append({
            'nom': 'Mildiou', 'type': 'Champignon',
            'risk': mildiou_risk,
            'niveau': 'danger' if mildiou_risk >= 65 else 'attention',
            'cause': f'Humidite {humidity}% + T {temperature}C favorables',
            'prevention': 'Maintenir hygrometrie basse, bien aerer, ne pas mouiller le feuillage.',
        })
        risk_max = max(risk_max, mildiou_risk)

    # 2. BOTRYTIS
    botrytis_risk = 0
    if humidity > 90 and 15 <= temperature <= 20:         botrytis_risk = 95
    elif humidity > 85 and 15 <= temperature <= 20:       botrytis_risk = 75
    elif humidity > 85 and 12 <= temperature <= 22:       botrytis_risk = 55
    elif humidity > 80 and 15 <= temperature <= 20:       botrytis_risk = 35
    if botrytis_risk > 0:
        diseases.append({
            'nom': 'Botrytis (Pourriture grise)', 'type': 'Champignon',
            'risk': botrytis_risk,
            'niveau': 'danger' if botrytis_risk >= 65 else 'attention',
            'cause': f'Humidite {humidity}% + T {temperature}C + faible ventilation',
            'prevention': 'Aeration quotidienne pour evacuer l\'humidite matinale.',
        })
        risk_max = max(risk_max, botrytis_risk)

    # 3. OIDIUM
    oidium_risk = 0
    if 40 <= humidity <= 80 and 20 <= temperature <= 27:   oidium_risk = 70
    elif 35 <= humidity <= 80 and 18 <= temperature <= 30: oidium_risk = 45
    if oidium_risk > 0:
        diseases.append({
            'nom': 'Oidium', 'type': 'Champignon',
            'risk': oidium_risk,
            'niveau': 'danger' if oidium_risk >= 65 else 'attention',
            'cause': f'Air sec ({humidity}%) + T chaude {temperature}C',
            'prevention': 'Eviter le stress hydrique, maintenir hygrometrie stable.',
        })
        risk_max = max(risk_max, oidium_risk)

    # 4. SCLEROTINIOSE
    sclerotiniose_risk = 0
    if sol > 80 and humidity > 85 and 10 <= temperature <= 20:   sclerotiniose_risk = 80
    elif sol > 70 and humidity > 80 and 10 <= temperature <= 20: sclerotiniose_risk = 55
    if sclerotiniose_risk > 0:
        diseases.append({
            'nom': 'Sclerotiniose (Pourriture blanche)', 'type': 'Champignon tellurique',
            'risk': sclerotiniose_risk,
            'niveau': 'danger' if sclerotiniose_risk >= 65 else 'attention',
            'cause': f'Sol sature ({sol}%) + humidite {humidity}% + T {temperature}C',
            'prevention': 'Rotation des cultures, drainage efficace du sol.',
        })
        risk_max = max(risk_max, sclerotiniose_risk)

    # 5. ACARIENS
    acariens_risk = 0
    if humidity < 65 and temperature > 25:    acariens_risk = 75
    elif humidity < 70 and temperature > 25:  acariens_risk = 50
    elif humidity < 70 and temperature > 22:  acariens_risk = 30
    if acariens_risk > 0:
        diseases.append({
            'nom': 'Acariens (araignees rouges)', 'type': 'Ravageur',
            'risk': acariens_risk,
            'niveau': 'danger' if acariens_risk >= 65 else 'attention',
            'cause': f'Air sec ({humidity}%) + T elevee {temperature}C',
            'prevention': 'Brumisations legeres pour augmenter l\'hygrometrie.',
        })
        risk_max = max(risk_max, acariens_risk)

    # 6. ALEURODES
    aleurodes_risk = 0
    if temperature > 25 and co2 > 1000:           aleurodes_risk = 65
    elif 20 <= temperature <= 30 and co2 > 800:   aleurodes_risk = 40
    elif 20 <= temperature <= 30:                 aleurodes_risk = 20
    if aleurodes_risk > 0:
        diseases.append({
            'nom': 'Aleurodes (Mouches blanches)', 'type': 'Ravageur',
            'risk': aleurodes_risk,
            'niveau': 'danger' if aleurodes_risk >= 65 else 'attention',
            'cause': f'T {temperature}C + serre confinee (CO2 {co2} ppm)',
            'prevention': 'Utilisation de pieges jaunes, ameliorer la ventilation.',
        })
        risk_max = max(risk_max, aleurodes_risk)

    # 7. THRIPS
    thrips_risk = 0
    if humidity < 50 and temperature > 20:    thrips_risk = 60
    elif humidity < 60 and temperature > 22:  thrips_risk = 40
    if thrips_risk > 0:
        diseases.append({
            'nom': 'Thrips', 'type': 'Ravageur',
            'risk': thrips_risk,
            'niveau': 'danger' if thrips_risk >= 65 else 'attention',
            'cause': f'Conditions seches et chaudes ({humidity}% / {temperature}C)',
            'prevention': 'Surveiller les fleurs. Augmenter l\'hygrometrie.',
        })
        risk_max = max(risk_max, thrips_risk)

    # 8. PUCERONS
    pucerons_risk = 0
    if 15 <= temperature <= 25 and humidity > 60:  pucerons_risk = 35
    elif 15 <= temperature <= 25:                  pucerons_risk = 20
    if pucerons_risk > 0:
        diseases.append({
            'nom': 'Pucerons', 'type': 'Ravageur',
            'risk': pucerons_risk, 'niveau': 'attention',
            'cause': f'T douce {temperature}C favorable a leur multiplication',
            'prevention': 'Eviter exces d\'engrais azotes, favoriser les auxiliaires.',
        })
        risk_max = max(risk_max, pucerons_risk)

    diseases.sort(key=lambda x: x['risk'], reverse=True)
    return diseases, risk_max


# ═════════════════════════════════════════════════════════════
# ROUTE 1 — Detection maladies
# ═════════════════════════════════════════════════════════════
@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    try:
        body        = request.get_json()
        temperature = float(body.get('temperature', 20))
        humidity    = float(body.get('humidity',    60))
        co2         = float(body.get('co2',         800))
        sol         = float(body.get('sol',         50))

        diseases, risk_max = detect_diseases(temperature, humidity, co2, sol)

        X     = np.array([[temperature, humidity]])
        pred  = rf_model.predict(X)[0]
        proba = rf_model.predict_proba(X)[0]

        labels         = ['bon', 'attention', 'danger']
        label          = labels[pred]
        prob_bon       = round(float(proba[0]) * 100, 1)
        prob_attention = round(float(proba[1]) * 100, 1) if len(proba) > 1 else 0.0
        prob_danger    = round(float(proba[2]) * 100, 1) if len(proba) > 2 else 0.0

        ml_risk  = round(prob_danger + prob_attention * 0.5)
        combined = max(ml_risk, risk_max)

        main_disease = diseases[0]['nom'] if diseases else 'Aucune'
        botrytis = next((d['risk'] for d in diseases if 'Botrytis' in d['nom']), 0)
        mildew   = next((d['risk'] for d in diseases if 'Mildiou'  in d['nom']), 0)

        return jsonify({
            'status':       'ok',
            'risk_level':   label,
            'risk_percent': combined,
            'risk':         combined,
            'botrytis':     botrytis,
            'mildew':       mildew,
            'disease':      main_disease,
            'diseases':     diseases,
            'ml_risk':      ml_risk,
            'probabilities': {
                'bon': prob_bon, 'attention': prob_attention, 'danger': prob_danger,
            }
        })
    except Exception as e:
        import traceback
        return jsonify({
            'status':    'error',
            'message':   str(e),
            'traceback': traceback.format_exc()
        }), 500


# ═════════════════════════════════════════════════════════════
# ROUTE 2 — Prevision LSTM temperature + humidity
# ═════════════════════════════════════════════════════════════
@app.route('/predict/lstm', methods=['POST'])
def predict_lstm():
    try:
        body        = request.get_json()
        current     = body.get('current', {})
        temperature = float(current.get('temperature', 20))
        humidity    = float(current.get('humidity',    60))

        now         = datetime.datetime.now()
        predictions = []

        if USE_LSTM and lstm_model is not None:
            seq = np.tile(
                np.array([[temperature, humidity]], dtype='float32'), (24, 1))
            seq += np.random.normal(0, 0.1, seq.shape).astype('float32')
            seq_norm = (seq - lstm_mean) / lstm_std
            seq_norm = seq_norm[np.newaxis, ...]

            for i in range(1, 7):
                pred_norm = lstm_model.predict(seq_norm, verbose=0)[0]
                temp_pred = round(float(
                    np.clip(pred_norm[0] * lstm_std[0] + lstm_mean[0], 10, 40)), 1)
                hum_pred  = int(
                    np.clip(pred_norm[1] * lstm_std[1] + lstm_mean[1], 30, 95))
                future_hour = now + datetime.timedelta(hours=i)
                predictions.append({
                    'label':       f'{future_hour.hour}h00',
                    'temperature': temp_pred,
                    'humidity':    hum_pred,
                })
                new_point = pred_norm[np.newaxis, np.newaxis, :]
                seq_norm  = np.concatenate(
                    [seq_norm[:, 1:, :], new_point], axis=1)
        else:
            for i in range(1, 7):
                temp_drift  = temp_coef[0] * i * 0.1
                temp_pred   = round(float(np.clip(
                    temperature + temp_drift + np.random.normal(0, 0.3), 10, 40)), 1)
                hum_delta   = -0.8 * (temp_pred - temperature)
                future_hour = now + datetime.timedelta(hours=i)
                predictions.append({
                    'label':       f'{future_hour.hour}h00',
                    'temperature': temp_pred,
                    'humidity':    int(np.clip(humidity + hum_delta, 30, 95)),
                })

        return jsonify({
            'status':      'ok',
            'predictions': predictions,
            'model':       'lstm' if USE_LSTM else 'leger',
        })
    except Exception as e:
        import traceback
        return jsonify({
            'status':    'error',
            'message':   str(e),
            'traceback': traceback.format_exc()
        }), 500


# ═════════════════════════════════════════════════════════════
# ROUTE 3 — Diagnostic plante (OpenRouter + Gemini Vision)
# ═════════════════════════════════════════════════════════════
@app.route('/predict/plant', methods=['POST'])
def predict_plant():
    try:
        print("=== /predict/plant appelé ===") 
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
                "model": "nvidia/nemotron-nano-12b-v2-vl:free",
                "messages": [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]}]
            },
            timeout=30
        )
        print("=== OpenRouter RAW ===")
        print("Status:", response.status_code)
        print("Body:", response.text[:1000])

        data  = response.json()
        text  = data['choices'][0]['message']['content']
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
            'status': 'error',
            'message': f'Reponse non parseable: {str(e)}',
            'raw': text if 'text' in locals() else ''
        }), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════
# ROUTE 4 — Conseil du jour
# ═════════════════════════════════════════════════════════════
@app.route('/predict/conseil', methods=['POST'])
def predict_conseil():
    if conseil_model is None or conseil_enc is None:
        return jsonify({
            'status':  'error',
            'message': 'Modele conseil non disponible.'
        }), 503

    try:
        body        = request.get_json()
        farmer_name = body.get('farmerName',  'Agriculteur')
        serre_name  = body.get('serreName',   'Ma Serre')
        temperature = float(body.get('temperature', 20))
        humidity    = float(body.get('humidity',    60))
        co2         = float(body.get('co2',         800))
        sol         = float(body.get('sol',         50))
        lumiere     = float(body.get('lumiere',     500))
        disease     = body.get('disease',     'Aucune')
        temp_max    = float(body.get('tempMax', temperature + 2))

        X            = np.array([[temperature, humidity, co2, lumiere, sol]])
        pred_encoded = conseil_model.predict(X)[0]
        proba        = conseil_model.predict_proba(X)[0]
        label        = conseil_enc.inverse_transform([pred_encoded])[0]
        confiance    = round(float(np.max(proba)) * 100)

        conseil_info = conseils_dict.get(label, conseils_dict['conditions_normales'])

        heure = datetime.datetime.now().strftime('%H:%M')
        conseil_text = (
            f"Bonjour {farmer_name} ! Il est {heure} et voici "
            f"votre analyse de {serre_name}.\n\n"
            f"{conseil_info['emoji']} {conseil_info['titre']}\n\n"
            f"Conditions actuelles :\n"
            f"- Temperature : {temperature}C (max prevue : {temp_max}C)\n"
            f"- Humidite : {humidity}%\n"
            f"- Luminosite : {lumiere} W/m2\n"
            f"- Sol : {sol}%  |  CO2 : {co2} ppm\n"
        )

        if disease != 'Aucune':
            conseil_text += f"- Risque detecte : {disease}\n"

        conseil_text += f"\nActions recommandees aujourd'hui :\n"
        for i, action in enumerate(conseil_info['actions'], 1):
            conseil_text += f"{i}. {action}\n"

        conseil_text += (
            f"\nDiagnostic IA : "
            f"{label.replace('_', ' ').title()} (confiance {confiance}%)"
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
        import traceback
        return jsonify({
            'status':    'error',
            'message':   str(e),
            'traceback': traceback.format_exc()
        }), 500


# ═════════════════════════════════════════════════════════════
# ROUTE 5 — Sync Firebase → CSV
# ═════════════════════════════════════════════════════════════
@app.route('/sync', methods=['POST'])
def sync_data():
    """Reçoit une mesure depuis Firebase et l'ajoute au CSV."""
    try:
        body = request.get_json()
        new_row = {
            'timestamp':   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': float(body.get('temperature', 20)),
            'humidity':    float(body.get('humidite', 60)),
            'lumiere':     float(body.get('Luminosite', 500)),
            'co2':         float(body.get('co2', 800)),
            'sol':         float(body.get('sol', 50)),
        }
        df = pd.read_csv('data/sensor_data.csv')
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv('data/sensor_data.csv', index=False)
        return jsonify({'status': 'ok', 'added': new_row})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Lancement ─────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
