from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Charger le modèle depuis le dossier local
classifier = pipeline("text-classification", model="./BERT-Emotions-Classifier")

# Créer une application FastAPI
app = FastAPI()

# Définir la structure des requêtes entrantes
class TextRequest(BaseModel):
    text: str

# Point d'entrée API
@app.post("/classify")
def classify_text(request: TextRequest):
    try:
        # Utiliser le modèle pour classifier le texte
        results = classifier(request.text)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
