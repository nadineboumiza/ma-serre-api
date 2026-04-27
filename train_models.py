import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
import joblib

os.makedirs('models', exist_ok=True)

print("Chargement des donnees...")
df = pd.read_csv('data/sensor_data.csv')
df = df.dropna()
print(f"{len(df)} lignes chargees")

# ═══════════════════════════════════════════════════════════════
# MODELE 1 — RANDOM FOREST
# Detecte le risque de maladie selon les seuils du PPTX
# Features : temperature + humidity uniquement (vrais capteurs)
# Labels   : 0=Bon  1=Attention  2=Danger
# ═══════════════════════════════════════════════════════════════
print("\nEntrainement Random Forest...")

def compute_risk(row):
    """
    Score de risque base sur les seuils du PPTX :
    Mildiou       : humidity > 90%  ET 10 <= temperature <= 25
    Botrytis      : humidity > 85%  ET 15 <= temperature <= 20
    Oidium        : 40 <= humidity <= 80 ET 20 <= temperature <= 27
    Sclerotiniose : humidity > 90%  ET 10 <= temperature <= 20
    Acariens      : humidity < 65%  ET temperature > 25
    Aleurodes     : 20 <= temperature <= 30
    Thrips        : temperature > 20 ET humidity < 50
    Pucerons      : 15 <= temperature <= 25
    """
    score = 0
    t = row['temperature']
    h = row['humidity']

    if h > 90 and 10 <= t <= 25:          score += 40   # Mildiou
    if h > 85 and 15 <= t <= 20:          score += 35   # Botrytis
    if 40 <= h <= 80 and 20 <= t <= 27:   score += 25   # Oidium
    if h > 90 and 10 <= t <= 20:          score += 30   # Sclerotiniose
    if h < 65 and t > 25:                 score += 35   # Acariens
    if 20 <= t <= 30:                     score += 15   # Aleurodes
    if t > 20 and h < 50:                 score += 25   # Thrips
    if 15 <= t <= 25:                     score += 10   # Pucerons

    if score >= 60: return 2    # Danger
    elif score >= 30: return 1  # Attention
    return 0                    # Bon

df['risk_label'] = df.apply(compute_risk, axis=1)

print("Distribution des classes :")
print(df['risk_label'].value_counts())

X_rf = df[['temperature', 'humidity']].values
y_rf = df['risk_label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)

print("\nRapport Random Forest :")
print(classification_report(
    y_test, rf.predict(X_test),
    target_names=['Bon', 'Attention', 'Danger'],
    zero_division=0,
))

joblib.dump(rf, 'models/rf_model.joblib')
print("Random Forest sauvegarde -> models/rf_model.joblib")

# ═══════════════════════════════════════════════════════════════
# MODELE 2 — LSTM
# Predit temperature ET humidity pour les 6 prochaines heures
# Input  : 24 mesures passees de [temperature, humidity]
# Output : [temperature_future, humidity_future]
# ═══════════════════════════════════════════════════════════════
print("\nEntrainement LSTM (temperature + humidity)...")

features  = ['temperature', 'humidity']
data_lstm = df[features].values.astype('float32')

# Normalisation
mean_vals = data_lstm.mean(axis=0)   # [temp_mean, hum_mean]
std_vals  = data_lstm.std(axis=0)    # [temp_std,  hum_std]
std_vals[std_vals == 0] = 1
data_norm = (data_lstm - mean_vals) / std_vals

np.save('models/lstm_mean.npy', mean_vals)
np.save('models/lstm_std.npy',  std_vals)

# Construction des sequences
# X : 24 mesures passees → y : 1 mesure future (temp + humidity)
SEQ_LEN = 24
X_seq, y_seq = [], []
for i in range(len(data_norm) - SEQ_LEN - 1):
    X_seq.append(data_norm[i:i + SEQ_LEN])    # (24, 2)
    y_seq.append(data_norm[i + SEQ_LEN])       # (2,)

X_seq = np.array(X_seq)   # shape (N, 24, 2)
y_seq = np.array(y_seq)   # shape (N, 2)
print(f"Sequences : X={X_seq.shape}  y={y_seq.shape}")

split      = int(len(X_seq) * 0.8)
X_tr, X_te = X_seq[:split], X_seq[split:]
y_tr, y_te = y_seq[:split], y_seq[split:]

# Architecture LSTM
model = keras.Sequential([
    keras.layers.Input(shape=(SEQ_LEN, 2)),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2),   # 2 sorties : temperature ET humidity
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

model.fit(
    X_tr, y_tr,
    epochs=30,
    batch_size=32,
    validation_data=(X_te, y_te),
    verbose=1,
)

loss, mae = model.evaluate(X_te, y_te, verbose=0)
print(f"\nLSTM — MAE : {mae:.4f}")

model.save('models/lstm_model.keras')
print("LSTM sauvegarde -> models/lstm_model.keras")

# ═══════════════════════════════════════════════════════════════
# MODELE LEGER — fallback si LSTM non disponible en production
# ═══════════════════════════════════════════════════════════════
print("\nExport modele leger...")

X_lin = np.arange(len(data_lstm)).reshape(-1, 1).astype('float32')
lr = LinearRegression()
lr.fit(X_lin, data_lstm[:, 0])

np.save('models/temp_coef.npy', np.array([lr.coef_[0], lr.intercept_]))

# Stats pour temperature et humidity uniquement
np.save('models/data_stats.npy', np.array([
    data_lstm[:, 0].mean(),   # temperature moyenne
    data_lstm[:, 0].std(),    # temperature ecart-type
    data_lstm[:, 1].mean(),   # humidity moyenne
    data_lstm[:, 1].std(),    # humidity ecart-type
]))

print("Modele leger sauvegarde !")
print("\nTous les modeles sont prets !")
print("  models/rf_model.joblib")
print("  models/lstm_model.keras")
print("  models/lstm_mean.npy  -> [temp_mean, hum_mean]")
print("  models/lstm_std.npy   -> [temp_std,  hum_std]")
print("  models/temp_coef.npy")
print("  models/data_stats.npy")