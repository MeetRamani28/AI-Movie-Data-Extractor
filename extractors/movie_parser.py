import os
import json
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 1. Page Configuration
st.set_page_config(
    page_title="CineParse AI - Premium Extractor", 
    page_icon="🎬", 
    layout="wide"
)

# 2. Fully Fixed & Proper Custom CSS
st.markdown("""
    <style>
    /* Main container spacing alignment */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Fixed Premium Gradient Title */
    .gradient-title {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8533 50%, #FFD000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0px !important;
        padding-bottom: 5px !important;
    }
    
    /* Subtitle Styling */
    .subtitle-text {
        color: #A0A0A0 !important;
        font-size: 1.15rem !important;
        margin-top: -5px !important;
        margin-bottom: 25px !important;
    }
    
    /* Modern Container Cards */
    .premium-card {
        background-color: #1E1E24 !important;
        border: 1px solid #2D2D34 !important;
        padding: 24px !important;
        border-radius: 16px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Section Headers Alignment */
    .section-header {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin-bottom: 15px !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Custom Decorative Badges for Genres & Cast */
    .custom-tag {
        background-color: rgba(255, 75, 75, 0.12) !important;
        color: #FF4B4B !important;
        padding: 6px 14px !important;
        border-radius: 50px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
        margin-right: 8px !important;
        margin-bottom: 8px !important;
        border: 1px solid rgba(255, 75, 75, 0.25) !important;
    }
    .cast-tag {
        background-color: rgba(255, 208, 0, 0.1) !important;
        color: #FFD000 !important;
        border: 1px solid rgba(255, 208, 0, 0.25) !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Pydantic Model Setup
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

# 4. App Header UI Render
st.markdown('<h1 class="gradient-title">🎬 CineParse AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Next-generation unstructured text to structured movie intelligence engine.</p>', unsafe_allow_html=True)

# 5. Core Layout Split (55% Left, 45% Right for perfect balance)
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.markdown('<div class="section-header">📥 Input Core</div>', unsafe_allow_html=True)
    
    # Textarea container
    para = st.text_area(
        "Paste Movie Paragraph, Review, or Meta-description:", 
        height=280, 
        placeholder="Interstellar is a visually stunning science fiction epic directed by Christopher Nolan. Released in 2014, the film stars Matthew McConaughey, Anne Hathaway...",
        label_visibility="collapsed" # UI ને કલીન રાખવા લેબલ હાઇડ કર્યું
    )
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    extract_btn = st.button("🚀 Run AI Extraction Engine", type="primary", use_container_width=True)

with col2:
    st.markdown('<div class="section-header">📤 Intelligence Output</div>', unsafe_allow_html=True)
    
    if extract_btn and para:
        with st.spinner("⚡ Processing via Mistral AI Neural Engine..."):
            try:
                # Prompt Setup
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
                json_data = json.loads(response.content)
                
                # Dynamic Tabs Display
                tab1, tab2 = st.tabs(["✨ Visual Dashboard", "💻 Raw JSON Data"])
                
                with tab1:
                    st.toast("Data Extracted Successfully!", icon="✅")
                    
                    # 3-Column Metrics Dashboard
                    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric(label="🎬 Movie Title", value=json_data.get("title", "N/A"))
                    m2.metric(label="📅 Release Year", value=json_data.get("release_year", "N/A"))
                    rating_val = f"⭐ {json_data.get('rating')}/10" if json_data.get('rating') else "N/A"
                    m3.metric(label="🏆 IMDb Rating", value=rating_val)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Movie Metadata Cards
                    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                    st.markdown(f"🎥 **Director:** &nbsp;&nbsp;`{json_data.get('director', 'Unknown')}`")
                    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
                    
                    # Rendering Genres Badges
                    st.markdown("**🏷️ Genres:**")
                    genres = json_data.get('genres', [])
                    if genres:
                        for g in genres:
                            st.markdown(f'<span class="custom-tag">{g}</span>', unsafe_allow_html=True)
                    else:
                        st.write("N/A")
                        
                    # Rendering Cast Badges
                    st.markdown("<div style='margin-top: 14px;'>**🎭 Starring Cast:**</div>", unsafe_allow_html=True)
                    cast = json_data.get('cast', [])
                    if cast:
                        for c in cast:
                            st.markdown(f'<span class="custom-tag cast-tag">{c}</span>', unsafe_allow_html=True)
                    else:
                        st.write("N/A")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Summary Highlight
                    st.markdown('<div class="premium-card" style="border-left: 4px solid #FF4B4B !important;">', unsafe_allow_html=True)
                    st.markdown("📝 **Executive Summary:**")
                    st.write(json_data.get("summary", "No summary extracted."))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.markdown("### JSON Response")
                    st.json(json_data)
                    
            except Exception as e:
                st.error(f"Extraction Pipeline Failed: {e}")
                
    elif extract_btn and not para:
        st.warning("🔮 System Input Empty: Please provide a text paragraph to trigger the engine.")
    else:
        # Default State Card
        st.markdown(
            '<div class="premium-card" style="color: #64B5F6; border: 1px dashed #2196F3 !important; background-color: rgba(33, 150, 243, 0.05) !important;">'
            '💡 <b>Standby Mode:</b> Enter a paragraph on the left and click extract to initialize AI processing.'
            '</div>', 
            unsafe_allow_html=True
        )