import tensorflow_datasets as tfds
import os
import numpy as np
from PIL import Image

print("📥 Téléchargement PlantVillage (~1.5 GB)...")
print("⏳ Cela peut prendre 10-20 minutes...")

# Téléchargement automatique via TensorFlow Datasets
ds, info = tfds.load(
    'plant_village',
    split         = ['train'],
    with_info     = True,
    as_supervised = True,
    data_dir      = 'data/tfds',
)

print(f"\n✅ Dataset téléchargé !")
print(f"   Nombre de classes : {info.features['label'].num_classes}")
print(f"   Noms classes      : {info.features['label'].names[:5]}...")

# ── Supprimer ancien dataset démo ─────────────────
import shutil
if os.path.exists('data/PlantVillage'):
    print("\n🗑️  Suppression du dataset démo...")
    shutil.rmtree('data/PlantVillage')
    print("✅ Dataset démo supprimé")

# ── Convertir en dossiers pour Keras ─────────────
print("\n🔄 Conversion en dossiers (peut prendre 5 min)...")

class_names = info.features['label'].names
OUT_DIR     = 'data/PlantVillage'

for name in class_names:
    os.makedirs(f'{OUT_DIR}/{name}', exist_ok=True)

counters = {name: 0 for name in class_names}
total    = 0

for img, label in tfds.as_numpy(ds[0]):
    class_name = class_names[label]
    count      = counters[class_name]
    path       = f'{OUT_DIR}/{class_name}/{count:05d}.jpg'
    Image.fromarray(img).save(path, quality=90)
    counters[class_name] += 1
    total += 1

    if total % 2000 == 0:
        print(f"   {total} images converties...")

print(f"\n✅ {total} images sauvegardées !")
print(f"   Classes : {len(class_names)}")
print("\nDistribution (5 premières classes) :")
for name, count in list(counters.items())[:5]:
    print(f"   {name}: {count} images")
print("\n🚀 Maintenant lancez : python train_plant_disease.py")