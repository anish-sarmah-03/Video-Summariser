import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
from google.generativeai import upload_file, get_file

import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)


st.set_page_config(page_title="Video Summariser AI Agent", page_icon="📺", layout="wide")
st.title("📺 Video Summariser AI Agent")

#header
st.markdown(
    """
    <h4 style='color: #a0aec0; font-weight: 400; margin-top: -15px; padding-bottom: 20px;'>
    Powered by Google Gemini ⚡
    </h4>
    """, 
    unsafe_allow_html=True
)

 #Sidebar
with st.sidebar:

    st.title("Info")
    st.markdown("This AI agent uses the **Gemini 3.1 Flash-Lite** model to analyze video content frame-by-frame and provide actionable insights.")
    st.divider()
    st.caption("Made using Streamlit & Phi-Data")



#CORE TECH LOGIC 
@st.cache_data
def initialize_agent():
    return Agent(
        name="Video Summariser Agent",
        model=Gemini(id="gemini-3.1-flash-lite-preview"),
        tools=[DuckDuckGo()],
        markdown=True,
    )
video_agent = initialize_agent()

video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"], help="Upload a video for its Analysis")

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    left, video_col, right = st.columns([1, 2, 1]) 
    
    with video_col:
        st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze it and provide you with the required information.",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    # Upload and process video file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # Prompt generation for analysis
                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}

                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )

                    # AI agent processing
                    response = video_agent.run(analysis_prompt, videos=[processed_video])

                #result
                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
else:
   #working
    st.markdown("---")
    st.subheader("💡 How it works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**1. Upload**\n\nDrag and drop your MP4, AVI, or MKV video file into the uploader.")
    with col2:
        st.info("**2. Ask**\n\nType in what you want to know(summaries, specific moments, or key insights).")
    with col3:
        st.success("**3. Analysis**\n\nOur AI agent will watch the video and generate detailed answers.")


#CSS styling
st.markdown(
    """
    <style>
    /* Gradient Main Title */
    h1 {
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Text Area custom height */
    .stTextArea textarea {
        height: 100px;
    }
    /* Fixed Video Size (added missing semicolons and !important) */
    /* Strictly force the Streamlit video container to shrink */
    div[data-testid="stVideo"] {
        max-width: 450px !important; /* Adjust this number to your liking */
        margin: 0 auto !important;   /* Centers the container */
        border-radius: 12px;
        overflow: hidden; /* Keeps the rounded corners clean */
        box-shadow: 0 4px 12px rgba(0,0,0,0.5); /* Optional: adds a nice shadow */
    }
    </style>
    """,
    unsafe_allow_html=True
)
#warning
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888888; font-size: 12px; padding: 10px;'>
    ⚠️ <b>Disclaimer:</b> AI can make mistakes. Please verify important information and insights with original sources.
    </div>
    """,
    unsafe_allow_html=True
)