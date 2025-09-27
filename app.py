import streamlit as st
from transformers import pipeline

# ==========================
# Page configuration
# ==========================
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================
# Custom CSS for styling
# ==========================
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color:white;
        font-size:16px;
        height:45px;
        width:100%;
        border-radius:10px;
        border:none;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border-radius:10px;
        padding:10px;
        font-size:16px;
    }
    h1 {
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Title and Description
# ==========================
st.title("📝 AI Text Summarizer")
st.markdown(
    """
    Paste any paragraph below, and our AI will generate a concise summary.
    Powered by **BART Large CNN** from Hugging Face Transformers.
    """
)

# ==========================
# Load summarization model
# ==========================
@st.cache_resource
def load_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = load_model()

# ==========================
# User input
# ==========================
text_input = st.text_area("Enter text here:", height=200)

# ==========================
# Summarize button
# ==========================
if st.button("Generate Summary"):
    if len(text_input.strip()) == 0:
        st.warning("⚠️ Please enter some text to summarize.")
    else:
        with st.spinner("Generating summary..."):
            input_words = len(text_input.split())
            max_len = min(250, max(120, int(input_words * 0.9)))
            min_len = max(80, int(max_len * 0.5))
            summary = summarizer(
                text_input, max_length=max_len, min_length=min_len, do_sample=False
            )
        st.subheader("✅ Summary")
        st.markdown(f"<div style='background-color:#ffffff;padding:15px;border-radius:10px;'>{summary[0]['summary_text']}</div>", unsafe_allow_html=True)

# ==========================
# Sidebar (Optional)
# ==========================
st.sidebar.header("About")
st.sidebar.info(
    """
    - Built with Streamlit and Hugging Face Transformers
    - Model: **facebook/bart-large-cnn**
    - Developed by You
    """
)
