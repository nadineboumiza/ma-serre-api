import pandas as pd
from datetime import datetime

print("🔄 Fusion des données...")

try:
    df_real = pd.read_csv('data/firebase_export.csv')
    df_real['timestamp'] = pd.to_datetime(
        df_real['timestamp'], utc=True, errors='coerce'
    )
    df_real['timestamp'] = df_real['timestamp'].dt.tz_localize(None)
    df_real['timestamp'] = df_real['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'co2' not in df_real.columns:
        df_real['co2'] = 800.0
    if 'sol' not in df_real.columns:
        df_real['sol'] = 50.0
    df_real = df_real[['timestamp','temperature','humidity','lumiere','co2','sol']]
    print(f"✅ Firebase réel : {len(df_real)} lignes")
except FileNotFoundError:
    print("⚠️ firebase_export.csv non trouvé")
    df_real = pd.DataFrame(columns=['timestamp','temperature','humidity','lumiere','co2','sol'])

try:
    df_generated = pd.read_csv('data/sensor_data.csv')
    df_generated['timestamp'] = pd.to_datetime(
        df_generated['timestamp'], errors='coerce'
    ).dt.strftime('%Y-%m-%d %H:%M:%S')
    print(f"✅ Données existantes : {len(df_generated)} lignes")
except FileNotFoundError:
    df_generated = pd.DataFrame(columns=['timestamp','temperature','humidity','lumiere','co2','sol'])

df_combined = pd.concat([df_generated, df_real], ignore_index=True)
df_combined.drop_duplicates(subset=['timestamp'], inplace=True)
df_combined.sort_values('timestamp', inplace=True)
df_combined.dropna(inplace=True)
df_combined.reset_index(drop=True, inplace=True)

print(f"✅ Dataset final : {len(df_combined)} lignes")
df_combined.to_csv('data/sensor_data.csv', index=False)
print("✅ sensor_data.csv mis à jour !")