import requests
import base64
import os

BASE_URL = "https://ma-serre-api.onrender.com" # ← remplacez par votre URL

print("=" * 50)
print("🧪 TEST API MA SERRE")
print("=" * 50)

# ── Route 1 — Random Forest ──────────────────────
print("\n📊 Test Route 1 — Random Forest...")
try:
    r1 = requests.post(f"{BASE_URL}/predict/disease", json={
        "temperature": 22,
        "humidity": 85,
        "co2": 1100,
        "sol": 60
    }, timeout=30)
    print(f"Status: {r1.status_code}")
    print(f"Réponse: {r1.json()}")
except Exception as e:
    print(f"❌ Erreur: {e}")

# ── Route 2 — LSTM ───────────────────────────────
print("\n🧠 Test Route 2 — LSTM...")
try:
    r2 = requests.post(f"{BASE_URL}/predict/lstm", json={
        "current": {
            "temperature": 22,
            "humidity": 65,
            "co2": 900,
            "lumiere": 20000,
            "sol": 50
        }
    }, timeout=120)  # ← 60s pour laisser le temps au modèle
    print(f"Status: {r2.status_code}")
    print(f"Réponse: {r2.text}")
except Exception as e:
    print(f"❌ Erreur: {e}")

# ── Route 3 — Gemini Vision ──────────────────────
print("\n🌿 Test Route 3 — Gemini Vision...")
if os.path.exists("test_plante.jpg"):
    try:
        with open("test_plante.jpg", "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        r3 = requests.post(f"{BASE_URL}/predict/plant", json={
            "image": image_b64,
            "media_type": "image/jpeg"
        }, timeout=30)
        print(f"Status: {r3.status_code}")
        print(f"Réponse: {r3.json()}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
else:
    print("⚠️  Ajoutez une photo nommée test_plante.jpg dans C:\\ma_serre_api\\")

print("\n" + "=" * 50)
print("✅ Tests terminés")
# ── Route 4 — Conseil ────────────────────────────
print("\n🤖 Test Route 4 — Conseil du jour...")
try:
    r4 = requests.post(f"{BASE_URL}/predict/conseil", json={
        "farmerName":  "Nadine",
        "serreName":   "SERRE1",
        "temperature": 22,
        "humidity":    65,
        "co2":         900,
        "sol":         50,
        "lumiere":     20000,
        "risk":        "bon",
        "disease":     "Aucune",
        "tempMax":     25
    }, timeout=30)
    print(f"Status: {r4.status_code}")
    print(f"Réponse: {r4.json()}")
except Exception as e:
    print(f"❌ Erreur: {e}")