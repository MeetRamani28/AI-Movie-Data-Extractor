import os
from dotenv import load_dotenv
from pydentic import BaseModel
from typing import Optional , List
from langchain_mistralai import ChatMistralAI

load_dotenv()

class Movie(BaseModel):
    title : str
    release_year: Optional[int] 
    genres: List[str] 
    director: Optional[str] 
    cast: List[str] 
    rating: Optional[float] 
    summary: str

api_key = os.getenv("MISTRAL_API_KEY")
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.0,
    mistral_api_key=api_key
)
print("Model and Schema Initialized Successfully!")