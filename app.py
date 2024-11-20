from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

# Initialiser FastAPI
app = FastAPI()

# Charger le pipeline de classification multi-label
classifier = pipeline(
    task="text-classification",
    model="ayoubkirouane/BERT-Emotions-Classifier",
    device=torch.cuda.current_device() if torch.cuda.is_available() else -1,
    top_k=None  # Retourner tous les scores
)

# Définir le schéma pour la requête
class TextRequest(BaseModel):
    text: str

# Route pour classifier un texte
@app.post("/classify")
def classify_text(request: TextRequest):
    # Texte à analyser
    text = request.text
    
    # Obtenir tous les scores pour toutes les étiquettes
    results = classifier(text)
    
    # Retourner les résultats bruts
    return {"classification": results[0]}
