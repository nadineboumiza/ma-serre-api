import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
docs = db.collection('mesures').stream()

rows = []
for doc in docs:
    d = doc.to_dict()
    rows.append({
        'timestamp':   d.get('date'),
        'temperature': d.get('temperature'),
        'humidity':    d.get('humidite'),
        'lumiere':     d.get('Luminosite'),
    })

df = pd.DataFrame(rows)
df = df.dropna()
df = df.sort_values('timestamp')
df.to_csv('data/firebase_export.csv', index=False)
print(f"✅ {len(df)} lignes exportées")
print(df.head(10))