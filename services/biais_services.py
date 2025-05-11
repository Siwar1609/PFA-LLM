#import spacy
#from collections import Counter
#import tensorflow as tf
#from transformers import CamembertTokenizer, TFCamembertForSequenceClassification
# services/pipeline_loader.py


# Chargements
#nlp_fr = spacy.load("fr_core_news_sm")
#sentiment_tokenizer = CamembertTokenizer.from_pretrained("tblard/tf-allocine")
#sentiment_model = TFCamembertForSequenceClassification.from_pretrained("tblard/tf-allocine")

#def lemmatize_text(text: str):
 #   doc = nlp_fr(text)
  #  return [token.lemma_.lower().strip() for token in doc if not token.is_stop and not token.is_punct]

#def detect_bias(text: str, stereotypes: list):
 #   lemmatized = lemmatize_text(text)
  #  counts = Counter(lemmatized)
   # bias_words = {word: counts[word] for word in stereotypes if word in counts}
    #total_found = sum(bias_words.values())
   # total_words = len(lemmatized)
   # score = (total_found / total_words) * 100 if total_words > 0 else 0
   # return bias_words, round(score, 2)

#def analyze_sentiment(text: str):
 #   inputs = sentiment_tokenizer(text, return_tensors="tf", padding=True, truncation=True)
 #   outputs = sentiment_model(inputs["input_ids"])
 #   probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
 #   labels = ["negative", "positive"]
 #   return labels[probs.argmax()].upper(), round(probs.max() * 100, 2)
# services/pipeline_loader.py

import os
from dotenv import load_dotenv
from transformers import pipeline

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer le token Hugging Face
hf_token = os.getenv("HF_AUTH_TOKEN")

# Vérifier que le token est présent
if not hf_token:
    raise ValueError("Le token Hugging Face est manquant. Assurez-vous que la variable d'environnement HF_AUTH_TOKEN est définie.")

# Charger les pipelines avec le token d'authentification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
