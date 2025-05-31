from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.ai_service import generate_with_gemini
from services.biais_services import classifier, sentiment_analyzer
from schemas import TextRequest

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle pour la génération de texte
class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"message": "Bienvenue dans notre app"}

@app.post("/generate")
async def generate_response(request: PromptRequest):
    result = await generate_with_gemini(request.prompt)
    if result.startswith("Erreur"):
        raise HTTPException(status_code=500, detail=result)
    return {"response": result}

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    sequence = request.text

    # Analyse des biais
    labels = [
        "religious bias", "ethnic bias", "gender bias",
        "age bias", "political bias", "not biased"
    ]
    bias_result = classifier(sequence, candidate_labels=labels)

    biases = [
        {"label": label, "score_percentage": round(score * 100, 2)}
        for label, score in zip(bias_result['labels'], bias_result['scores'])
    ]

    # Analyse du sentiment
    sentiment_result = sentiment_analyzer(sequence)
    sentiment_label_raw = sentiment_result[0]['label']
    sentiment_score = sentiment_result[0]['score']

    if sentiment_label_raw in ["1 star", "2 stars"]:
        sentiment = "Negative"
    elif sentiment_label_raw == "3 stars":
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    sentiment_result_clean = {
        "sentiment": sentiment,
        "confidence_percentage": round(sentiment_score * 100, 2)
    }

    return {
        "biases": biases,
        "sentiment": sentiment_result_clean
    }