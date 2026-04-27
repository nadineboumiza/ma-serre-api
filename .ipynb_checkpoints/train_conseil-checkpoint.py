import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

os.makedirs('models', exist_ok=True)

print("📊 Chargement des données...")
df = pd.read_csv('data/sensor_data.csv')
df = df.dropna()
print(f"✅ {len(df)} lignes chargées")

# ═══════════════════════════════════════════════════
# ÉTAPE 1 — Générer les conseils selon règles expertes
# Seuils alignés sur seuil_et_bioagresseur.pptx
# ═══════════════════════════════════════════════════
print("\n📝 Génération des conseils experts...")

def generate_conseil_label(row):
    """
    Règles expertes pour classifier la situation.
    Seuils alignés sur le PPTX et detect_diseases() de app.py.

    ─── SEUILS PPTX ──────────────────────────────────────────
    Mildiou       : h > 90% ET 10°C ≤ T ≤ 25°C  → risque_maladie
    Botrytis      : h > 85% ET 15°C ≤ T ≤ 20°C  → risque_botrytis
    Oïdium        : 40% ≤ h ≤ 80% ET 20°C ≤ T ≤ 27°C → risque_oidium
    Sclérotiniose : h > 90% ET 10°C ≤ T ≤ 20°C  → humidite_critique
    Acariens      : h < 65% ET T > 25°C          → humidite_faible
    Aleurodes     : 20°C ≤ T ≤ 30°C              → chaleur_moderee
    Thrips        : T > 20°C ET h < 50%           → humidite_faible
    Pucerons      : 15°C ≤ T ≤ 25°C              → conditions_normales (risque faible)
    ──────────────────────────────────────────────────────────
    """
    t      = row['temperature']
    h      = row['humidity']
    co2    = row['co2']
    sol    = row['sol']
    lum    = row['lumiere']  # Étape 4 : utilisé pour lumiere_faible

    # ── Danger immédiat ──────────────────────────────────────
    if t > 35:
        return 'chaleur_extreme'
    if t < 10:
        return 'froid_extreme'

    # CORRIGÉ (ex h>88) : aligné sur seuil Mildiou/Sclérotiniose PPTX (h>90)
    if h > 90:
        return 'humidite_critique'

    if sol < 20:
        return 'sol_sec_urgent'

    # ── Attention ────────────────────────────────────────────
    if t > 30 and h > 75:
        return 'chaud_humide'
    if t > 30:
        return 'chaleur_moderee'

    # CORRIGÉ (ex h>80 T 15-25) : aligné sur seuil Mildiou PPTX (h>85 T 10-25)
    if h > 85 and 10 <= t <= 25:
        return 'risque_maladie'

    # CORRIGÉ (ex h>75 T 18-22) : aligné sur seuil Botrytis PPTX (h>85 T 15-20)
    if h > 85 and 15 <= t <= 20:
        return 'risque_botrytis'

    # Oïdium PPTX : air SEC + T° chaude (contrairement aux autres champignons)
    if 40 <= h <= 80 and 20 <= t <= 27:
        return 'risque_oidium'

    # Acariens / Thrips PPTX : sec et chaud
    if h < 65 and t > 25:
        return 'humidite_faible'
    if t > 20 and h < 50:
        return 'humidite_faible'

    if co2 > 1200:
        return 'co2_eleve'
    if sol < 35:
        return 'sol_sec'
    if h < 40:
        return 'humidite_faible'

    # Étape 4 : lumiere utilisée comme signal
    if lum < 50 and 7 <= t :
        return 'lumiere_faible'

    # ── Optimal ──────────────────────────────────────────────
    if 18 <= t <= 28 and 50 <= h <= 75 and sol >= 50:
        return 'conditions_optimales'

    return 'conditions_normales'


df['conseil_label'] = df.apply(generate_conseil_label, axis=1)

print("\nDistribution des conseils :")
print(df['conseil_label'].value_counts())

# ═══════════════════════════════════════════════════
# ÉTAPE 2 — Encoder les labels
# ═══════════════════════════════════════════════════
le = LabelEncoder()
df['conseil_encoded'] = le.fit_transform(df['conseil_label'])

joblib.dump(le, 'models/conseil_encoder.joblib')
print(f"\n✅ {len(le.classes_)} classes : {list(le.classes_)}")

# ═══════════════════════════════════════════════════
# ÉTAPE 3 — Entraîner le modèle
# ═══════════════════════════════════════════════════
print("\n🌲 Entraînement modèle conseil...")

X = df[['temperature', 'humidity', 'co2', 'lumiere', 'sol']].values
y = df['conseil_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

conseil_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
)
conseil_model.fit(X_train, y_train)

print("\n📊 Rapport :")
print(classification_report(
    y_test,
    conseil_model.predict(X_test),
    target_names=le.classes_,
    zero_division=0,
))

joblib.dump(conseil_model, 'models/conseil_model.joblib')
print("✅ Modèle conseil sauvegardé → models/conseil_model.joblib")

# ═══════════════════════════════════════════════════
# ÉTAPE 4 — Dictionnaire de conseils textuels
# ═══════════════════════════════════════════════════
conseils_dict = {
    'chaleur_extreme': {
        'emoji':   '🔥',
        'titre':   'Chaleur extrême détectée !',
        'actions': [
            'Ouvrez immédiatement toutes les fenêtres et aérations',
            'Activez le système de refroidissement ou ventilateurs',
            'Arrosez les allées pour rafraîchir l\'air ambiant',
            'Protégez les plantes fragiles avec un voile d\'ombrage',
            'Surveillez les plantes toutes les 30 minutes',
        ],
        'urgence': 'critique',
    },
    'froid_extreme': {
        'emoji':   '🥶',
        'titre':   'Température trop basse !',
        'actions': [
            'Fermez toutes les ouvertures immédiatement',
            'Activez le chauffage de la serre',
            'Couvrez les plantes sensibles avec un voile thermique',
            'Vérifiez les plants les plus fragiles',
            'Évitez tout arrosage jusqu\'au réchauffement',
        ],
        'urgence': 'critique',
    },
    'humidite_critique': {
        'emoji':   '💧',
        'titre':   'Humidité critique — risque Mildiou / Sclérotiniose !',
        'actions': [
            'Augmentez la ventilation immédiatement (seuil PPTX : h > 90%)',
            'Ouvrez les fenêtres et activez les ventilateurs',
            'Évitez tout arrosage aujourd\'hui',
            'Inspectez les feuilles pour taches de mildiou ou pourriture blanche',
            'Appliquez un fongicide préventif si nécessaire',
        ],
        'urgence': 'elevee',
    },
    'sol_sec_urgent': {
        'emoji':   '🏜️',
        'titre':   'Sol très sec — arrosage urgent !',
        'actions': [
            'Arrosez abondamment toutes les plantes maintenant',
            'Vérifiez que le système d\'irrigation fonctionne',
            'Augmentez la fréquence d\'arrosage cette semaine',
            'Ajoutez du paillis pour retenir l\'humidité du sol',
            'Surveillez l\'état des plantes après arrosage',
        ],
        'urgence': 'elevee',
    },
    'chaud_humide': {
        'emoji':   '🌫️',
        'titre':   'Conditions chaudes et humides',
        'actions': [
            'Améliorez la circulation d\'air dans la serre',
            'Réduisez l\'arrosage de 30% aujourd\'hui',
            'Inspectez les feuilles pour signes de maladie',
            'Appliquez un traitement préventif contre le mildiou',
            'Surveillez la température toutes les heures',
        ],
        'urgence': 'moyenne',
    },
    'chaleur_moderee': {
        'emoji':   '☀️',
        'titre':   'Température élevée',
        'actions': [
            'Ouvrez les aérations pour ventiler la serre',
            'Arrosez tôt le matin ou en soirée',
            'Vérifiez l\'ombrage des plantes sensibles',
            'Maintenez le sol humide mais pas détrempé',
            'Surveillez les Aleurodes (mouches blanches) si T° > 25°C',
        ],
        'urgence': 'faible',
    },
    'risque_maladie': {
        'emoji':   '🦠',
        'titre':   'Risque Mildiou — humidité > 85% et T° favorable',
        'actions': [
            'Augmentez la ventilation dès maintenant',
            'Inspectez toutes les feuilles pour taches ou moisissures',
            'Évitez tout arrosage par aspersion aujourd\'hui',
            'Appliquez un fongicide préventif sur les plantes',
            'Éliminez les feuilles malades si trouvées',
        ],
        'urgence': 'elevee',
    },
    'risque_botrytis': {
        'emoji':   '🍂',
        'titre':   'Risque de Botrytis — h > 85% et 15–20°C',
        'actions': [
            'Réduisez l\'humidité en ventilant la serre (seuil PPTX : h > 85%)',
            'Inspectez les fruits et fleurs pour pourriture grise',
            'Supprimez immédiatement les parties infectées',
            'Appliquez un traitement fongicide spécifique Botrytis',
            'Évitez de mouiller le feuillage lors de l\'arrosage',
        ],
        'urgence': 'elevee',
    },
    # NOUVEAU : Oïdium ajouté (manquait dans l'ancien dictionnaire)
    'risque_oidium': {
        'emoji':   '⬜',
        'titre':   'Risque d\'Oïdium — air sec et T° chaude',
        'actions': [
            'Vérifiez les feuilles pour dépôts blancs poudreux (seuil PPTX : 40–80% / 20–27°C)',
            'Évitez le stress hydrique des plantes',
            'Maintenez une hygrométrie stable (éviter les écarts)',
            'Appliquez un traitement soufré préventif si nécessaire',
            'Réduisez les courants d\'air sec dans la serre',
        ],
        'urgence': 'moyenne',
    },
    'co2_eleve': {
        'emoji':   '💨',
        'titre':   'Taux de CO₂ élevé',
        'actions': [
            'Ouvrez les fenêtres pour renouveler l\'air',
            'Vérifiez le bon fonctionnement de la ventilation',
            'Contrôlez les équipements de chauffage',
            'Augmentez la durée de ventilation quotidienne',
        ],
        'urgence': 'moyenne',
    },
    'sol_sec': {
        'emoji':   '🌱',
        'titre':   'Sol un peu sec',
        'actions': [
            'Arrosez modérément les plantes ce soir',
            'Vérifiez l\'humidité du sol à 5cm de profondeur',
            'Ajustez le programme d\'irrigation si automatique',
            'Évitez d\'arroser en pleine chaleur',
        ],
        'urgence': 'faible',
    },
    'humidite_faible': {
        'emoji':   '🏜️',
        'titre':   'Air trop sec — risque Acariens / Thrips',
        'actions': [
            'Augmentez l\'arrosage progressivement',
            'Vaporisez de l\'eau sur les allées pour augmenter l\'hygrométrie',
            'Inspectez le dessous des feuilles pour araignées rouges (seuil PPTX : h < 65%)',
            'Surveillez les fleurs pour présence de Thrips (h < 50%)',
            'Installez un humidificateur si disponible',
        ],
        'urgence': 'faible',
    },
    # NOUVEAU : lumiere_faible ajouté (étape 4 — signal lumiere utilisé)
    'lumiere_faible': {
        'emoji':   '🌑',
        'titre':   'Luminosité insuffisante',
        'actions': [
            'Vérifiez si les panneaux de la serre sont propres',
            'Nettoyez les vitrages pour maximiser la lumière naturelle',
            'Envisagez un éclairage d\'appoint si nécessaire',
            'Évitez d\'arroser en excès par temps sombre (favorise les maladies)',
        ],
        'urgence': 'faible',
    },
    'conditions_optimales': {
        'emoji':   '✅',
        'titre':   'Conditions excellentes !',
        'actions': [
            'Continuez votre programme d\'arrosage habituel',
            'Profitez pour faire une inspection générale des plantes',
            'C\'est le bon moment pour fertiliser si nécessaire',
            'Vérifiez la croissance et notez les progrès',
            'Maintenez ces conditions le plus longtemps possible',
        ],
        'urgence': 'aucune',
    },
    'conditions_normales': {
        'emoji':   '🌿',
        'titre':   'Conditions normales',
        'actions': [
            'Maintenez votre routine d\'entretien habituelle',
            'Arrosez selon les besoins de chaque plante',
            'Faites une inspection visuelle des feuilles',
            'Vérifiez les niveaux de nutriments si nécessaire',
        ],
        'urgence': 'aucune',
    },
}

# Sauvegarder le dictionnaire
import json
with open('models/conseils_dict.json', 'w',
          encoding='utf-8') as f:
    json.dump(conseils_dict, f,
              ensure_ascii=False, indent=2)

print("✅ Dictionnaire conseils sauvegardé → models/conseils_dict.json")
print("\n🎉 Modèle conseil prêt !")
print("📁 models/conseil_model.joblib")
print("📁 models/conseil_encoder.joblib")
print("📁 models/conseils_dict.json")