import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional , List
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

class Movie(BaseModel):
    title : str
    release_year: Optional[int] 
    genres: List[str] 
    director: Optional[str] 
    cast: List[str] 
    rating: Optional[float] 
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract The Following Information About The Movie From the Paragraph.\n{format_instructions}"),
    ("human", "{paragraph}")
]
)

para = input("Enter Paragraph : ")

final_prompt = prompt.format_prompt(
    **{
        "paragraph": para,
        "format_instructions": parser.get_format_instructions()
    }
)

api_key = os.getenv("MISTRAL_API_KEY")
model = ChatMistralAI(model="mistral-small-latest", temperature=0.0, mistral_api_key=api_key)

response = model.invoke(final_prompt.to_messages())
print("\n--- Extracted JSON Output ---")
print(response.content)