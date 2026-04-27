"""
generate_data.py
Genere 5000 lignes de donnees realistes pour une serre tunisienne.

Colonnes :
- timestamp   : date et heure (intervalle 5 minutes)
- temperature : degres Celsius
- humidity    : pourcentage %
- lumiere     : W/m2 (pyranometre, plage 0-2000 W/m2)
- co2         : ppm (valeur par defaut 800, pas de capteur)
- sol         : % (valeur par defaut 50, pas de capteur)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

os.makedirs('data', exist_ok=True)
np.random.seed(42)

N          = 5000
INTERVAL   = 5                          # minutes entre chaque mesure
START_DATE = datetime(2025, 4, 1, 0, 0, 0)

print("Generation des donnees serre tunisienne...")

rows = []
current_time = START_DATE

for i in range(N):
    hour   = current_time.hour
    minute = current_time.minute
    day    = current_time.day
    h_dec  = hour + minute / 60.0      # heure decimale ex: 14h30 = 14.5

    # ── TEMPERATURE (°C) ─────────────────────────────────────
    # Tunisie avril : 18°C la nuit, 28°C le jour
    # Minimum a 5h du matin, maximum a 14h
    temp_base = 23 + 5 * np.sin(np.pi * (h_dec - 5) / 12)
    temp_base += day * 0.05            # tendance montante printemps
    if 8 <= hour <= 18:
        temp_base += 3 * np.sin(np.pi * (h_dec - 8) / 10)  # effet serre
    temperature = round(float(np.clip(
        temp_base + np.random.normal(0, 0.4), 12, 38)), 1)

    # ── HUMIDITE (%) ─────────────────────────────────────────
    # Inverse de la temperature : haute la nuit (75-85%), basse le jour (50-65%)
    hum_base = 80 - 20 * np.sin(np.pi * (h_dec - 5) / 12)
    if 6 <= hour <= 9:
        hum_base += 8 * np.sin(np.pi * (h_dec - 6) / 3)   # condensation matin
    if 7 <= hour <= 8:
        hum_base += 10                 # arrosage matin
    if 16 <= hour <= 17:
        hum_base += 8                  # arrosage soir
    humidity = round(float(np.clip(
        hum_base + np.random.normal(0, 2), 35, 95)), 1)

    # ── LUMIERE pyranometre (W/m2) ───────────────────────────
    # Plage capteur : 0 a 2000 W/m2
    # Tunisie avril ciel clair : pic ~850 W/m2 a midi
    # Nuit : 0 W/m2
    if 6 <= hour <= 20:
        lumiere_base = 850 * np.exp(-0.5 * ((h_dec - 12.5) / 3.5) ** 2)
        if np.random.random() < 0.2:   # nuages 20% du temps
            lumiere_base *= np.random.uniform(0.1, 0.5)
        lumiere = round(float(np.clip(
            lumiere_base + np.random.normal(0, 15), 0, 2000)), 1)
    else:
        lumiere = 0.0                  # nuit

    rows.append({
        'timestamp':   current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'temperature': temperature,
        'humidity':    humidity,
        'lumiere':     lumiere,        # W/m2
        'co2':         800.0,          # valeur par defaut
        'sol':         50.0,           # valeur par defaut
    })

    current_time += timedelta(minutes=INTERVAL)

# ── Sauvegarder ──────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv('data/sensor_data.csv', index=False)

print(f"{len(df)} lignes sauvegardees dans data/sensor_data.csv")
print(f"Du {df['timestamp'].iloc[0]} au {df['timestamp'].iloc[-1]}")
print("\nStatistiques :")
print(df[['temperature', 'humidity', 'lumiere']].describe().round(1))

df_day   = df[df['timestamp'].str[11:13].astype(int).between(8, 18)]
df_night = df[df['timestamp'].str[11:13].astype(int) < 6]
print(f"\nJour  — T: {df_day['temperature'].mean():.1f}C  "
      f"H: {df_day['humidity'].mean():.1f}%  "
      f"Lum: {df_day['lumiere'].mean():.0f} W/m2")
print(f"Nuit  — T: {df_night['temperature'].mean():.1f}C  "
      f"H: {df_night['humidity'].mean():.1f}%  "
      f"Lum: {df_night['lumiere'].mean():.0f} W/m2")

print("\nProchaines etapes :")
print("  1. python train_models.py")
print("  2. python train_conseil.py")
print("  3. python app.py")