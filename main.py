from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
#from services.biais_services import detect_bias, analyze_sentiment
from services.ai_service import generate_with_gemini
#from stereotypes import gender_stereotypes
from schemas import TextRequest
from services.biais_services import classifier, sentiment_analyzer

app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_response(request: PromptRequest):
    result = await generate_with_gemini(request.prompt)
    if result.startswith("Erreur"):
        raise HTTPException(status_code=500, detail=result)
    return {"response": result}

@app.get("/")
def root():
    return {"message": "Bienvenue dans notre app"}
#@app.post("/analyze")
#async def analyze_text(request: TextRequest):
   # male_bias, male_score = detect_bias(request.text, gender_stereotypes["male"])
    #female_bias, female_score = detect_bias(request.text, gender_stereotypes["female"])
    #sentiment_label, sentiment_score = analyze_sentiment(request.text)

    #result = {
     #   "male_bias": male_bias,
      #  "male_score": float(male_score),
       # "female_bias": female_bias,
        #"female_score": float(female_score),
        #"sentiment_label": sentiment_label,
       # "sentiment_score": float(sentiment_score),
        #"bias_type": (
         #   "masculine" if male_score > female_score else
          #  "feminine" if female_score > male_score else "neutral"
        #)
    #}

    #return result 

@app.post("/analyze")
async def analyze_text(request: TextRequest):
    sequence = request.text

    # Analyse des biais
    labels = ["religious bias", "ethnic bias", "gender bias", "age bias", "political bias", "not biased"]
    bias_result = classifier(sequence, candidate_labels=labels)

    biases = []
    for label, score in zip(bias_result['labels'], bias_result['scores']):
        biases.append({
            "label": label,
            "score_percentage": round(score * 100, 2)
        })

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