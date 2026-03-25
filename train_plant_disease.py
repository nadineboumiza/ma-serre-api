import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

print("🌿 Entraînement MobileNetV2 — PlantVillage RÉEL")
print("=" * 55)

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS_H   = 10
EPOCHS_F   = 15
DATA_DIR   = 'data/PlantVillage'

# ═══════════════════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════════════════
print(f"\n📂 Chargement depuis {DATA_DIR} ...")

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split = 0.2,
    subset           = 'training',
    seed             = 42,
    image_size       = (IMG_SIZE, IMG_SIZE),
    batch_size       = BATCH_SIZE,
    label_mode       = 'categorical',
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split = 0.2,
    subset           = 'validation',
    seed             = 42,
    image_size       = (IMG_SIZE, IMG_SIZE),
    batch_size       = BATCH_SIZE,
    label_mode       = 'categorical',
)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"✅ {NUM_CLASSES} classes — "
      f"{sum(1 for _ in train_ds) * BATCH_SIZE} "
      f"images train")

os.makedirs('models', exist_ok=True)
with open('models/plant_classes.json', 'w',
          encoding='utf-8') as f:
    json.dump(CLASS_NAMES, f,
              ensure_ascii=False, indent=2)
print("✅ plant_classes.json sauvegardé")

# ═══════════════════════════════════════════════════
# DICTIONNAIRE MALADIES → CONSEILS AGRICULTEUR
# ═══════════════════════════════════════════════════
DISEASE_INFO = {}
for cls in CLASS_NAMES:
    plante = cls.split('___')[0].replace('_', ' ') \
        if '___' in cls \
        else cls.split('_')[0]

    if 'healthy' in cls.lower():
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   'Aucune',
            'statut':    'saine',
            'urgence':   False,
            'cause':     'Plante en bonne santé.',
            'symptomes': ['Feuilles vertes normales',
                          'Pas de taches visibles'],
            'traitement': ['Continuer l\'arrosage régulier',
                           'Surveillance normale',
                           'Maintenir ventilation'],
            'prevention': 'Continuez vos bonnes pratiques !',
        }
    elif any(k in cls.lower() for k in
             ['late_blight', 'late blight']):
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   'Mildiou tardif',
            'statut':    'critique',
            'urgence':   True,
            'cause':     'Phytophthora infestans — '
                         'très destructeur et contagieux.',
            'symptomes': ['Taches brunes irrégulières',
                          'Bords jaunâtres',
                          'Pourriture rapide'],
            'traitement': ['TRAITEMENT FONGICIDE URGENT',
                           'Isoler les plants infectés',
                           'Détruire parties malades'],
            'prevention': 'Variétés résistantes '
                          'et traitements préventifs.',
        }
    elif any(k in cls.lower() for k in
             ['early_blight', 'early blight']):
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   'Alternariose (mildiou précoce)',
            'statut':    'malade',
            'urgence':   False,
            'cause':     'Champignon Alternaria solani — '
                         'humidité et chaleur.',
            'symptomes': ['Taches brunes concentriques',
                          'Jaunissement des feuilles',
                          'Défoliation progressive'],
            'traitement': ['Retirer feuilles infectées',
                           'Fongicide cuivré',
                           'Réduire humidité'],
            'prevention': 'Rotation des cultures '
                          'et arrosage à la base.',
        }
    elif 'leaf_mold' in cls.lower() or \
         'leaf mold' in cls.lower():
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   'Moisissure des feuilles',
            'statut':    'malade',
            'urgence':   False,
            'cause':     'Champignon Passalora fulva — '
                         'humidité > 85%.',
            'symptomes': ['Taches jaunâtres dessus',
                          'Moisissure grisâtre dessous'],
            'traitement': ['Améliorer ventilation',
                           'Réduire humidité < 80%',
                           'Fongicide si nécessaire'],
            'prevention': 'Humidité < 80% '
                          'et aération régulière.',
        }
    elif 'bacterial_spot' in cls.lower() or \
         'bacterial spot' in cls.lower():
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   'Tache bactérienne',
            'statut':    'attention',
            'urgence':   False,
            'cause':     'Bactérie Xanthomonas — '
                         'eau contaminée.',
            'symptomes': ['Petites taches noires',
                          'Bordure jaune',
                          'Chute des feuilles'],
            'traitement': ['Traitement cuivré',
                           'Arrosage à la base',
                           'Éliminer débris'],
            'prevention': 'Eau propre '
                          'et éviter blessures.',
        }
    elif 'mosaic' in cls.lower() or \
         'virus' in cls.lower():
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   'Virus mosaïque',
            'statut':    'critique',
            'urgence':   True,
            'cause':     'Virus transmis par pucerons — '
                         'pas de traitement curatif.',
            'symptomes': ['Feuilles mosaïquées',
                          'Déformation',
                          'Croissance ralentie'],
            'traitement': ['Arracher plants infectés',
                           'Traiter contre pucerons',
                           'Désinfecter outils'],
            'prevention': 'Contrôle insectes '
                          'et semences certifiées.',
        }
    elif 'spider_mite' in cls.lower() or \
         'spider mite' in cls.lower():
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   'Acarien (araignée rouge)',
            'statut':    'attention',
            'urgence':   False,
            'cause':     'Tétranyque urticae — '
                         'chaleur et sécheresse.',
            'symptomes': ['Petits points jaunes',
                          'Toiles fines',
                          'Feuilles bronzées'],
            'traitement': ['Acaricide biologique',
                           'Augmenter humidité',
                           'Pulvérisation eau froide'],
            'prevention': 'Maintenir humidité '
                          'et surveiller en été.',
        }
    elif 'rust' in cls.lower():
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   'Rouille',
            'statut':    'malade',
            'urgence':   True,
            'cause':     'Champignon Puccinia — '
                         'se propage par le vent.',
            'symptomes': ['Pustules orangées',
                          'Poudre rougeâtre',
                          'Feuilles qui sèchent'],
            'traitement': ['Fongicide soufré',
                           'Retirer feuilles',
                           'Traitement hebdomadaire'],
            'prevention': 'Espacement plants '
                          'et éviter excès azote.',
        }
    elif 'scab' in cls.lower():
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   'Gale',
            'statut':    'attention',
            'urgence':   False,
            'cause':     'Champignon Venturia — '
                         'printemps humide.',
            'symptomes': ['Taches liégeuses',
                          'Fruits déformés',
                          'Feuilles tachées'],
            'traitement': ['Fongicide préventif',
                           'Ramasser feuilles tombées',
                           'Tailler parties atteintes'],
            'prevention': 'Variétés résistantes '
                          'et traitements au printemps.',
        }
    elif 'powdery_mildew' in cls.lower() or \
         'powdery mildew' in cls.lower():
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   'Oïdium',
            'statut':    'malade',
            'urgence':   False,
            'cause':     'Champignon — air chaud et sec.',
            'symptomes': ['Poudre blanche sur feuilles',
                          'Jaunissement',
                          'Déformation'],
            'traitement': ['Bicarbonate de soude',
                           'Fongicide soufré',
                           'Améliorer ventilation'],
            'prevention': 'Bonne circulation d\'air '
                          'et arrosage le matin.',
        }
    else:
        DISEASE_INFO[cls] = {
            'plante':    plante,
            'maladie':   cls.replace('_', ' ')
                           .replace('___', ' — '),
            'statut':    'attention',
            'urgence':   False,
            'cause':     'Anomalie détectée.',
            'symptomes': ['Symptômes visibles sur feuille'],
            'traitement': ['Isoler le plant',
                           'Consulter un agronome',
                           'Surveiller évolution'],
            'prevention': 'Surveillance régulière.',
        }

with open('models/disease_info.json', 'w',
          encoding='utf-8') as f:
    json.dump(DISEASE_INFO, f,
              ensure_ascii=False, indent=2)
print("✅ disease_info.json sauvegardé")

# ═══════════════════════════════════════════════════
# PIPELINE OPTIMISÉ
# ═══════════════════════════════════════════════════
AUTOTUNE = tf.data.AUTOTUNE

augmentation = keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.15),
])

def preprocess(img, label):
    return tf.cast(img, tf.float32) / 255.0, label

train_ds = (
    train_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .map(lambda x, y:
         (augmentation(x, training=True), y),
         num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(2000)
    .prefetch(AUTOTUNE)
)
val_ds = (
    val_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(AUTOTUNE)
)

# ═══════════════════════════════════════════════════
# MODÈLE MobileNetV2
# ═══════════════════════════════════════════════════
print("\n🧠 Construction MobileNetV2...")

base = MobileNetV2(
    input_shape = (IMG_SIZE, IMG_SIZE, 3),
    include_top = False,
    weights     = 'imagenet',
)
base.trainable = False

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x   = base(inp, training=False)
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dense(512, activation='relu')(x)
x   = layers.Dropout(0.4)(x)
x   = layers.Dense(256, activation='relu')(x)
x   = layers.Dropout(0.3)(x)
out = layers.Dense(
        NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inp, out)
model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy'],
)
print(f"✅ Modèle prêt — {NUM_CLASSES} classes")

# ═══════════════════════════════════════════════════
# PHASE 1 — Tête seulement
# ═══════════════════════════════════════════════════
print(f"\n🔥 Phase 1 — {EPOCHS_H} epochs (tête)...")

h1 = model.fit(
    train_ds,
    epochs          = EPOCHS_H,
    validation_data = val_ds,
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor              = 'val_accuracy',
            patience             = 4,
            restore_best_weights = True,
        ),
    ],
    verbose = 1,
)
acc1 = max(h1.history['val_accuracy'])
print(f"\n✅ Phase 1 — Val accuracy : {acc1:.1%}")

# ═══════════════════════════════════════════════════
# PHASE 2 — Fine-tuning
# ═══════════════════════════════════════════════════
print(f"\n🔓 Phase 2 — Fine-tuning {EPOCHS_F} epochs...")

base.trainable = True
for layer in base.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer = keras.optimizers.Adam(1e-5),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy'],
)

h2 = model.fit(
    train_ds,
    epochs          = EPOCHS_F,
    validation_data = val_ds,
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor              = 'val_accuracy',
            patience             = 5,
            restore_best_weights = True,
        ),
        keras.callbacks.ModelCheckpoint(
            'models/plant_disease_model.keras',
            monitor        = 'val_accuracy',
            save_best_only = True,
            verbose        = 1,
        ),
    ],
    verbose = 1,
)
acc2 = max(h2.history['val_accuracy'])

# ═══════════════════════════════════════════════════
# RÉSULTAT
# ═══════════════════════════════════════════════════
model.save('models/plant_disease_model.keras')

print("\n" + "=" * 55)
print("🎉 ENTRAÎNEMENT TERMINÉ !")
print(f"   Phase 1 accuracy : {acc1:.1%}")
print(f"   Phase 2 accuracy : {acc2:.1%}")
print(f"   Classes          : {NUM_CLASSES}")
print("   Fichiers :")
print("   ├── models/plant_disease_model.keras")
print("   ├── models/plant_classes.json")
print("   └── models/disease_info.json")
print("=" * 55)