import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import joblib

os.makedirs('models', exist_ok=True)

print("Chargement des donnees...")
df = pd.read_csv('data/sensor_data.csv')
df = df.dropna()
print(f"{len(df)} lignes chargees")

# ═══════════════════════════════════════════════════
# ETAPE 1 — Generer les labels selon regles expertes
# Seuils alignes sur seuil_et_bioagresseur.pptx
# Lumiere en W/m2 (pyranometre, plage 0-2000 W/m2)
# ═══════════════════════════════════════════════════
print("\nGeneration des labels experts...")

def generate_conseil_label(row):
    t   = row['temperature']
    h   = row['humidity']
    co2 = row['co2']
    sol = row['sol']
    lum = row['lumiere']                      # W/m2
    # Extraire l'heure depuis le timestamp pour detecter le jour
    heure = int(str(row['timestamp'])[11:13])

    # ── Danger immediat ──────────────────────────────────────
    if t > 35:
        return 'chaleur_extreme'
    if t < 10:
        return 'froid_extreme'
    if h > 90:                                # seuil PPTX Mildiou/Sclerotiniose
        return 'humidite_critique'
    if sol < 20:
        return 'sol_sec_urgent'

    # ── Attention ────────────────────────────────────────────
    if t > 30 and h > 75:
        return 'chaud_humide'
    if t > 30:
        return 'chaleur_moderee'
    if h > 85 and 10 <= t <= 25:              # seuil PPTX Mildiou
        return 'risque_maladie'
    if h > 85 and 15 <= t <= 20:              # seuil PPTX Botrytis
        return 'risque_botrytis'
    if 40 <= h <= 80 and 20 <= t <= 27:       # seuil PPTX Oidium
        return 'risque_oidium'
    if h < 65 and t > 25:                     # seuil PPTX Acariens
        return 'humidite_faible'
    if t > 20 and h < 50:                     # seuil PPTX Thrips
        return 'humidite_faible'
    if co2 > 1200:
        return 'co2_eleve'
    if sol < 35:
        return 'sol_sec'
    if h < 40:
        return 'humidite_faible'

    # lumiere_faible : jour (8h-18h) mais moins de 50 W/m2 (nuageux)
    if lum < 50 and 8 <= heure <= 18:
        return 'lumiere_faible'

    # ── Optimal ──────────────────────────────────────────────
    if 18 <= t <= 28 and 50 <= h <= 75 and sol >= 50:
        return 'conditions_optimales'

    return 'conditions_normales'


df['conseil_label'] = df.apply(generate_conseil_label, axis=1)

print("\nDistribution des conseils :")
print(df['conseil_label'].value_counts())

# ═══════════════════════════════════════════════════
# ETAPE 2 — Encoder les labels
# ═══════════════════════════════════════════════════
le = LabelEncoder()
df['conseil_encoded'] = le.fit_transform(df['conseil_label'])

joblib.dump(le, 'models/conseil_encoder.joblib')
print(f"\n{len(le.classes_)} classes : {list(le.classes_)}")

# ═══════════════════════════════════════════════════
# ETAPE 3 — Entrainer le modele
# Features : temperature, humidity, co2, lumiere, sol
# ═══════════════════════════════════════════════════
print("\nEntrainement modele conseil...")

X = df[['temperature', 'humidity', 'co2', 'lumiere', 'sol']].values
y = df['conseil_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

conseil_model = RandomForestClassifier(
    n_estimators=100, random_state=42, max_depth=10)
conseil_model.fit(X_train, y_train)

print("\nRapport :")
print(classification_report(
    y_test, conseil_model.predict(X_test),
    target_names=le.classes_, zero_division=0,
))

joblib.dump(conseil_model, 'models/conseil_model.joblib')
print("Modele conseil sauvegarde -> models/conseil_model.joblib")

# ═══════════════════════════════════════════════════
# ETAPE 4 — Dictionnaire de conseils textuels
# ═══════════════════════════════════════════════════
conseils_dict = {
    'chaleur_extreme': {
        'emoji': '🔥',
        'titre': 'Chaleur extreme detectee !',
        'actions': [
            'Ouvrez immediatement toutes les fenetres et aerations',
            'Activez le systeme de refroidissement ou ventilateurs',
            'Arrosez les allees pour rafraichir l\'air ambiant',
            'Protegez les plantes fragiles avec un voile d\'ombrage',
            'Surveillez les plantes toutes les 30 minutes',
        ],
        'urgence': 'critique',
    },
    'froid_extreme': {
        'emoji': '🥶',
        'titre': 'Temperature trop basse !',
        'actions': [
            'Fermez toutes les ouvertures immediatement',
            'Activez le chauffage de la serre',
            'Couvrez les plantes sensibles avec un voile thermique',
            'Verifiez les plants les plus fragiles',
            'Evitez tout arrosage jusqu\'au rechauffement',
        ],
        'urgence': 'critique',
    },
    'humidite_critique': {
        'emoji': '💧',
        'titre': 'Humidite critique — risque Mildiou / Sclerotiniose !',
        'actions': [
            'Augmentez la ventilation immediatement (humidite > 90%)',
            'Ouvrez les fenetres et activez les ventilateurs',
            'Evitez tout arrosage aujourd\'hui',
            'Inspectez les feuilles pour taches de mildiou ou pourriture blanche',
            'Appliquez un fongicide preventif si necessaire',
        ],
        'urgence': 'elevee',
    },
    'sol_sec_urgent': {
        'emoji': '🏜️',
        'titre': 'Sol tres sec — arrosage urgent !',
        'actions': [
            'Arrosez abondamment toutes les plantes maintenant',
            'Verifiez que le systeme d\'irrigation fonctionne',
            'Augmentez la frequence d\'arrosage cette semaine',
            'Ajoutez du paillis pour retenir l\'humidite du sol',
            'Surveillez l\'etat des plantes apres arrosage',
        ],
        'urgence': 'elevee',
    },
    'chaud_humide': {
        'emoji': '🌫️',
        'titre': 'Conditions chaudes et humides',
        'actions': [
            'Ameliorez la circulation d\'air dans la serre',
            'Reduisez l\'arrosage de 30% aujourd\'hui',
            'Inspectez les feuilles pour signes de maladie',
            'Appliquez un traitement preventif contre le mildiou',
            'Surveillez la temperature toutes les heures',
        ],
        'urgence': 'moyenne',
    },
    'chaleur_moderee': {
        'emoji': '☀️',
        'titre': 'Temperature elevee',
        'actions': [
            'Ouvrez les aerations pour ventiler la serre',
            'Arrosez tot le matin ou en soiree',
            'Verifiez l\'ombrage des plantes sensibles',
            'Maintenez le sol humide mais pas detrempe',
            'Surveillez les Aleurodes (mouches blanches) si T > 25C',
        ],
        'urgence': 'faible',
    },
    'risque_maladie': {
        'emoji': '🦠',
        'titre': 'Risque Mildiou — humidite > 85% et T favorable',
        'actions': [
            'Augmentez la ventilation des maintenant',
            'Inspectez toutes les feuilles pour taches ou moisissures',
            'Evitez tout arrosage par aspersion aujourd\'hui',
            'Appliquez un fongicide preventif sur les plantes',
            'Eliminez les feuilles malades si trouvees',
        ],
        'urgence': 'elevee',
    },
    'risque_botrytis': {
        'emoji': '🍂',
        'titre': 'Risque de Botrytis — humidite > 85% et 15-20C',
        'actions': [
            'Reduisez l\'humidite en ventilant la serre',
            'Inspectez les fruits et fleurs pour pourriture grise',
            'Supprimez immediatement les parties infectees',
            'Appliquez un traitement fongicide specifique Botrytis',
            'Evitez de mouiller le feuillage lors de l\'arrosage',
        ],
        'urgence': 'elevee',
    },
    'risque_oidium': {
        'emoji': '⬜',
        'titre': 'Risque d\'Oidium — air sec et T chaude',
        'actions': [
            'Verifiez les feuilles pour depots blancs poudreux',
            'Evitez le stress hydrique des plantes',
            'Maintenez une hygometrie stable',
            'Appliquez un traitement soufre preventif si necessaire',
            'Reduisez les courants d\'air sec dans la serre',
        ],
        'urgence': 'moyenne',
    },
    'co2_eleve': {
        'emoji': '💨',
        'titre': 'Taux de CO2 eleve',
        'actions': [
            'Ouvrez les fenetres pour renouveler l\'air',
            'Verifiez le bon fonctionnement de la ventilation',
            'Controlez les equipements de chauffage',
            'Augmentez la duree de ventilation quotidienne',
        ],
        'urgence': 'moyenne',
    },
    'sol_sec': {
        'emoji': '🌱',
        'titre': 'Sol un peu sec',
        'actions': [
            'Arrosez moderement les plantes ce soir',
            'Verifiez l\'humidite du sol a 5cm de profondeur',
            'Ajustez le programme d\'irrigation si automatique',
            'Evitez d\'arroser en pleine chaleur',
        ],
        'urgence': 'faible',
    },
    'humidite_faible': {
        'emoji': '🏜️',
        'titre': 'Air trop sec — risque Acariens / Thrips',
        'actions': [
            'Augmentez l\'arrosage progressivement',
            'Vaporisez de l\'eau sur les allees pour augmenter l\'hygrometrie',
            'Inspectez le dessous des feuilles pour araignees rouges',
            'Surveillez les fleurs pour presence de Thrips',
            'Installez un humidificateur si disponible',
        ],
        'urgence': 'faible',
    },
    'lumiere_faible': {
        'emoji': '🌑',
        'titre': 'Luminosite insuffisante (< 50 W/m2)',
        'actions': [
            'Verifiez si les panneaux de la serre sont propres',
            'Nettoyez les vitrages pour maximiser la lumiere naturelle',
            'Envisagez un eclairage d\'appoint si necessaire',
            'Evitez d\'arroser en exces par temps sombre',
        ],
        'urgence': 'faible',
    },
    'conditions_optimales': {
        'emoji': '✅',
        'titre': 'Conditions excellentes !',
        'actions': [
            'Continuez votre programme d\'arrosage habituel',
            'Profitez pour faire une inspection generale des plantes',
            'C\'est le bon moment pour fertiliser si necessaire',
            'Verifiez la croissance et notez les progres',
            'Maintenez ces conditions le plus longtemps possible',
        ],
        'urgence': 'aucune',
    },
    'conditions_normales': {
        'emoji': '🌿',
        'titre': 'Conditions normales',
        'actions': [
            'Maintenez votre routine d\'entretien habituelle',
            'Arrosez selon les besoins de chaque plante',
            'Faites une inspection visuelle des feuilles',
            'Verifiez les niveaux de nutriments si necessaire',
        ],
        'urgence': 'aucune',
    },
}

with open('models/conseils_dict.json', 'w', encoding='utf-8') as f:
    json.dump(conseils_dict, f, ensure_ascii=False, indent=2)

print("Dictionnaire conseils sauvegarde -> models/conseils_dict.json")
print("\nModele conseil pret !")
print("  models/conseil_model.joblib")
print("  models/conseil_encoder.joblib")
print("  models/conseils_dict.json")