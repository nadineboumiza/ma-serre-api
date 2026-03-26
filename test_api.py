import requests
import base64
import json

# L'URL de ton serveur Flask local
URL = "http://127.0.0.1:5000/predict/plant"
# Le nom de ton image
IMAGE_FILE = "test_plante.jpg" 

def lancer_test():
    try:
        # 1. Conversion de l'image en texte (Base64)
        with open(IMAGE_FILE, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # 2. Préparation du paquet (JSON)
        payload = {
            "image": image_data,
            "media_type": "image/jpeg"
        }

        # 3. Envoi au serveur Flask
        print("📡 Envoi de la photo à l'API Gemini...")
        reponse = requests.post(URL, json=payload)

        # 4. Lecture du résultat
        if reponse.status_code == 200:
            print("✅ DIAGNOSTIC REÇU :")
            print(json.dumps(reponse.json(), indent=4, ensure_ascii=False))
        else:
            print(f"❌ Erreur {reponse.status_code} : {reponse.text}")

    except FileNotFoundError:
        print(f"⚠️ Erreur : Le fichier {IMAGE_FILE} est introuvable.")
    except Exception as e:
        print(f"⚠️ Une erreur est survenue : {e}")

if __name__ == "__main__":
    lancer_test()