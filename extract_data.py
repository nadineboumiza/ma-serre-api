import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
from datetime import datetime

# ── Initialiser Firebase ──────────────────────────────
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://pfe2026-146d0-default-rtdb.firebaseio.com'
})

# ── Lire les données depuis Firebase ──────────────────
print("📡 Connexion à Firebase Realtime Database...")
ref  = db.reference('/')
data = ref.get()
print(f"✅ Données reçues : {data}")

# ── Construire les lignes réelles ─────────────────────
rows = []
if isinstance(data, dict):
    rows.append({
        'temperature': float(data.get('temperature', 0)),
        'humidity':    float(data.get('humidite',    0)),
        'co2':         float(data.get('co2',         800)),
        'lumiere':     float(data.get('lumiere',     20000)),
        'sol':         float(data.get('sol',         50)),
    })

print(f"📊 {len(rows)} lecture(s) récupérée(s) depuis Firebase.")
print("🔄 Génération de données simulées réalistes...")

# ── Générer données simulées ──────────────────────────
n     = 1000
hours = pd.date_range('2025-01-01', periods=n, freq='5min')

temperature = 20 + 8 * np.sin(
    2 * np.pi * hours.hour / 24) + np.random.normal(0, 1.5, n)
humidity = 65 - 15 * np.sin(
    2 * np.pi * hours.hour / 24) + np.random.normal(0, 4, n)
co2 = 800 + 200 * np.sin(
    2 * np.pi * hours.hour / 24) + np.random.normal(0, 60, n)
lumiere = 30000 * np.maximum(
    0, np.sin(np.pi * (hours.hour - 6) / 12)
) + np.random.normal(0, 500, n)
sol = 55 + np.random.normal(0, 6, n)

df = pd.DataFrame({
    'timestamp':   hours,
    'temperature': np.clip(temperature, 15, 45),
    'humidity':    np.clip(humidity,    30, 95),
    'co2':         np.clip(co2,         400, 2000),
    'lumiere':     np.clip(lumiere,     0,   80000),
    'sol':         np.clip(sol,         20,  90),
})

# ── Ajouter les vraies lectures Firebase ──────────────
if rows:
    real_df = pd.DataFrame(rows)
    real_df['timestamp'] = datetime.now().isoformat()
    real_df['lumiere']   = 20000.0
    df = pd.concat([real_df, df], ignore_index=True)

# ── Sauvegarder ───────────────────────────────────────
df.to_csv('data/sensor_data.csv', index=False)
print(f"✅ {len(df)} lignes sauvegardées dans data/sensor_data.csv")
print(df.head(10))