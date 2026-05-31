import os
import json
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

st.set_page_config(
    page_title="CineParse AI", 
    page_icon="🎬", 
    layout="centered"
)

st.markdown("""
    <style>
    /* Main container layout */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 800px !important;
    }
    
    /* Gemini-inspired Premium Gradient Title */
    .gemini-title {
        background: linear-gradient(90deg, #4285F4 0%, #9B51E0 50%, #EA4335 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 2px !important;
        letter-spacing: -1px;
    }
    
    .subtitle-text {
        color: #70757A !important;
        font-size: 1.15rem !important;
        margin-top: 0px !important;
        margin-bottom: 30px !important;
    }
    
    /* Cross-Theme Adaptive Container Cards */
    .gemini-card {
        background-color: rgba(66, 133, 244, 0.04) !important;
        border: 1px solid rgba(128, 134, 139, 0.2) !important;
        padding: 22px !important;
        border-radius: 16px !important;
        margin-top: 15px !important;
        margin-bottom: 15px !important;
    }
    
    /* Fixed Title Color for both themes */
    .movie-display-title {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 0px !important;
        padding: 0px !important;
    }
    
    .section-label {
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        color: #1A73E8 !important;
        font-weight: 700 !important;
        margin-bottom: 12px !important;
    }
    
    /* Adaptive Badges for Tags */
    .custom-tag {
        background-color: rgba(60, 64, 67, 0.08) !important;
        color: #3C4043 !important;
        padding: 6px 14px !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
        margin-right: 8px !important;
        margin-bottom: 8px !important;
        border: 1px solid rgba(60, 64, 67, 0.15) !important;
    }
    
    /* Dark mode override helper for tags if user switches theme */
    @media (prefers-color-scheme: dark) {
        .custom-tag {
            background-color: rgba(240, 244, 249, 0.1) !important;
            color: #E8EAED !important;
            border: 1px solid rgba(240, 244, 249, 0.2) !important;
        }
        .movie-display-title {
            color: #FFFFFF !important;
        }
    }
    
    /* Modern Pill Button */
    div.stButton > button:first-child {
        border-radius: 50px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        background: linear-gradient(90deg, #4285F4, #9B51E0) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 2px 6px rgba(66, 133, 244, 0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

class Movie(BaseModel):
    title: str
    release_year: Optional[int] 
    genres: List[str] 
    director: Optional[str] 
    cast: List[str] 
    rating: Optional[float] 
    summary: str

load_dotenv()
parser = PydanticOutputParser(pydantic_object=Movie)

st.markdown('<h1 class="gemini-title">🎬 CineParse AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Transform complex movie paragraphs into structured intelligence.</p>', unsafe_allow_html=True)

para = st.text_area(
    "Enter movie text:", 
    height=180, 
    placeholder="Type or paste your movie details here (e.g., Interstellar is a 2014 sci-fi movie directed by Christopher Nolan...)",
    label_visibility="collapsed"
)

st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
extract_btn = st.button("🚀 Analyze Text with Neural Engine")
st.markdown("---")

if extract_btn and para:
    with st.spinner("⚡ Gemini Engine is thinking..."):
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract the following information about the movie from the paragraph.\n{format_instructions}"),
                ("human", "{paragraph}")
            ])
            
            final_prompt = prompt.format_prompt(
                **{
                    "paragraph": para,
                    "format_instructions": parser.get_format_instructions()
                }
            )
            
            model = ChatMistralAI(model="mistral-small-latest", temperature=0.0, mistral_api_key=os.getenv("MISTRAL_API_KEY"))
            response = model.invoke(final_prompt.to_messages())
            
            parsed_output = parser.parse(response.content)
            json_data = parsed_output.model_dump()
            
            tab1, tab2 = st.tabs(["✨ Intelligence Dashboard", "💻 Clean JSON Node"])
            
            with tab1:
                st.toast("Analysis Completed!", icon="✨")
                
                st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
                title_text = json_data.get("title", "N/A").upper()
                year_text = f"({json_data.get('release_year')})" if json_data.get('release_year') else ""
                rating_text = f"⭐ {json_data.get('rating')}/10" if json_data.get('rating') else "Rating: N/A"
                
                st.markdown(f"<div class='movie-display-title'>{title_text} <span style='color:#70757A; font-size:1.4rem; font-weight:400;'>{year_text}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#E67E22; font-weight:700; margin-top:6px; font-size:1.05rem;'>{rating_text}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="gemini-card" style="border-left: 4px solid #4285F4 !important;">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">📝 Executive Summary</div>', unsafe_allow_html=True)
                st.markdown(f"<div style='line-height:1.6;'>{json_data.get('summary', 'N/A')}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="gemini-card">', unsafe_allow_html=True)
                st.markdown(f"🎬 **Director:** &nbsp;&nbsp;`{json_data.get('director', 'Unknown')}`")
                st.markdown("<div style='margin-bottom: 18px;'></div>", unsafe_allow_html=True)
                
                st.markdown('<div class="section-label">🏷️ Genres</div>', unsafe_allow_html=True)
                genres = json_data.get('genres', [])
                if genres:
                    for g in genres:
                        st.markdown(f'<span class="custom-tag">{g}</span>', unsafe_allow_html=True)
                else:
                    st.write("N/A")
                    
                st.markdown("<div style='margin-bottom: 18px;'></div>", unsafe_allow_html=True)
                
                st.markdown('<div class="section-label">🎭 Starring Cast</div>', unsafe_allow_html=True)
                cast = json_data.get('cast', [])
                if cast:
                    for c in cast:
                        st.markdown(f'<span class="custom-tag">{c}</span>', unsafe_allow_html=True)
                else:
                    st.write("N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
                st.json(json_data)
                
        except Exception as e:
            st.error(f"AI Extraction Pipeline Interrupted: {e}")
            
elif extract_btn and not para:
    st.warning("🔮 Please input a movie description first to trigger the core engine.")
else:
    st.markdown("<p style='text-align: center; color: #70757A; font-size: 0.95rem;'>Ready to decode. Enter movie details above and click run.</p>", unsafe_allow_html=True)