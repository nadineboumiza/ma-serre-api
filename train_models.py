import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import joblib

print("📊 Chargement des données...")
df = pd.read_csv('data/sensor_data.csv')
df = df.dropna()
print(f"✅ {len(df)} lignes chargées")

# ═══════════════════════════════════════════════════
# MODÈLE 1 — RANDOM FOREST (Risque de maladie)
# ═══════════════════════════════════════════════════
print("\n🌲 Entraînement Random Forest...")

def compute_risk(row):
    risk = 0
    if row['humidity'] > 80 and 15 <= row['temperature'] <= 25:
        risk += 40
    if row['humidity'] > 85:
        risk += 20
    if row['co2'] > 1200:
        risk += 10
    if row['humidity'] > 75 and 18 <= row['temperature'] <= 22:
        risk += 25
    if row['temperature'] > 33:
        risk += 15
    if risk > 50:
        return 2   # danger
    elif risk > 25:
        return 1   # attention
    return 0       # bon

df['risk_label'] = df.apply(compute_risk, axis=1)

print("Distribution des classes :")
print(df['risk_label'].value_counts())

X_rf = df[['temperature', 'humidity', 'co2', 'sol']].values
y_rf = df['risk_label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("\n📊 Rapport Random Forest :")
print(classification_report(
    y_test, y_pred,
    target_names=['Bon', 'Attention', 'Danger'],
    zero_division=0,
))

joblib.dump(rf, 'models/rf_model.joblib')
print("✅ Random Forest sauvegardé → models/rf_model.joblib")

# ═══════════════════════════════════════════════════
# MODÈLE 2 — LSTM (Prévision température 6h)
# ═══════════════════════════════════════════════════
print("\n🧠 Entraînement LSTM...")

features  = ['temperature', 'humidity', 'co2', 'lumiere', 'sol']
data_lstm = df[features].values.astype('float32')

mean_vals = data_lstm.mean(axis=0)
std_vals  = data_lstm.std(axis=0)
std_vals[std_vals == 0] = 1
data_norm = (data_lstm - mean_vals) / std_vals

np.save('models/lstm_mean.npy', mean_vals)
np.save('models/lstm_std.npy',  std_vals)

SEQ_LEN = 24
X_seq, y_seq = [], []

for i in range(len(data_norm) - SEQ_LEN - 1):
    X_seq.append(data_norm[i:i + SEQ_LEN])
    y_seq.append(data_norm[i + SEQ_LEN, 0])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"Séquences créées : {X_seq.shape}")

split      = int(len(X_seq) * 0.8)
X_tr, X_te = X_seq[:split],  X_seq[split:]
y_tr, y_te = y_seq[:split],  y_seq[split:]

model = keras.Sequential([
    keras.layers.LSTM(
        64,
        return_sequences=True,
        input_shape=(SEQ_LEN, len(features)),
    ),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1),
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae'],
)

model.summary()

history = model.fit(
    X_tr, y_tr,
    epochs=30,
    batch_size=32,
    validation_data=(X_te, y_te),
    verbose=1,
)

loss, mae = model.evaluate(X_te, y_te, verbose=0)
print(f"\n📊 LSTM — MAE : {mae:.4f}")

model.save('models/lstm_model.keras')
print("✅ LSTM sauvegardé → models/lstm_model.keras")

print("\n🎉 Entraînement terminé !")
print("📁 Modèles sauvegardés dans models/")