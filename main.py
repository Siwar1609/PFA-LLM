from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from services.ai_service import generate_with_gemini
from schemas import TextRequest
from services.biais_services import classifier, sentiment_analyzer
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import databases

DATABASE_URL = "sqlite:///./prompts.db"

database = databases.Database(DATABASE_URL)
metadata = databases.Database(DATABASE_URL)
Base = declarative_base()

# Modèle de données
class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    prompt_text = Column(Text, nullable=False)
    response_text = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    bias_analysis = Column(Text)
    sentiment_analysis = Column(Text)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dépendance pour obtenir la session de la base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_response(request: PromptRequest, db: Session = Depends(get_db)):
    result = await generate_with_gemini(request.prompt)
    if result.startswith("Erreur"):
        raise HTTPException(status_code=500, detail=result)
    
    # Sauvegarder le prompt et la réponse
    db_prompt = Prompt(
        prompt_text=request.prompt,
        response_text=result
    )
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    
    return {"response": result, "prompt_id": db_prompt.id}

@app.post("/analyze")
async def analyze_text(request: TextRequest, db: Session = Depends(get_db)):
    sequence = request.text

    # Analyse des biais (conservée comme dans votre version initiale)
    labels = ["religious bias", "ethnic bias", "gender bias", "age bias", "political bias", "not biased"]
    bias_result = classifier(sequence, candidate_labels=labels)

    biases = []
    for label, score in zip(bias_result['labels'], bias_result['scores']):
        biases.append({
            "label": label,
            "score_percentage": round(score * 100, 2)
        })

    # Analyse du sentiment (conservée comme dans votre version initiale)
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

    # Sauvegarder l'analyse dans la base de données
    db_prompt = Prompt(
        prompt_text=sequence,
        bias_analysis=str(biases),
        sentiment_analysis=str(sentiment_result_clean)
    )
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)

    return {
        "biases": biases,
        "sentiment": sentiment_result_clean,
        "analysis_id": db_prompt.id
    }

@app.get("/prompts/")
async def get_prompts(db: Session = Depends(get_db)):
    prompts = db.query(Prompt).order_by(Prompt.created_at.desc()).all()
    return prompts

@app.get("/prompts/{prompt_id}")
async def get_prompt(prompt_id: int, db: Session = Depends(get_db)):
    prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt
@app.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: int, db: Session = Depends(get_db)):
    prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    db.delete(prompt)
    db.commit()
    return {"message": "Prompt deleted successfully"}
@app.get("/")
def root():
    return {"message": "Bienvenue dans notre app"}