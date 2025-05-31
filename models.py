from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from database import Base

class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    prompt_text = Column(Text, nullable=False)
    response_text = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    bias_analysis = Column(Text)  
    sentiment_analysis = Column(Text)  

# Créez les tables dans la base de données
def init_db():
    Base.metadata.create_all(bind=database.engine)