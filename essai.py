from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Charger le modèle et le tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Créer l'application FastAPI
app = FastAPI()

# Définir le modèle de données pour la requête
class QuestionRequest(BaseModel):
    context: str
    question: str

# Fonction pour obtenir la réponse à partir du modèle
def get_answer(context: str, question: str) -> str:
    # Tokenizer les entrées
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    
    # Effectuer l'inférence avec le modèle
    with torch.no_grad():
        outputs = model(**inputs)

    # Extraire la réponse
    answer_start = torch.argmax(outputs.start_logits)  # Index du début de la réponse
    answer_end = torch.argmax(outputs.end_logits) + 1  # Index de la fin de la réponse
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    return answer

# Route pour interroger l'API
@app.post("/answer/")
async def answer_question(request: QuestionRequest):
    answer = get_answer(request.context, request.question)
    return {"answer": answer}
