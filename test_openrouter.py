import requests
import base64

OPENROUTER_KEY = "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"  # ← votre vraie clé

with open("test_plante.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "openrouter/free",  # ✅ routeur automatique gratuit
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": "Dis juste bonjour"}
        ]}]
    }
)

print(r.status_code)
print(r.json())