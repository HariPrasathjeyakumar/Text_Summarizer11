import streamlit as st
from transformers import pipeline

# ==========================
# Page configuration
# ==========================
st.set_page_config(
    page_title="📝 AI Text Summarizer",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================
# Custom CSS for dark theme
# ==========================
st.markdown("""
    <style>
    /* App background */
    .stApp {
        background-color: #1f2937;
        color: #f9fafb;
    }

    /* Title */
    h1 {
        color: #facc15;
        text-align: center;
        font-size: 40px;
    }

    /* Description */
    .css-1d391kg p {
        color: #e5e7eb;
        font-size: 18px;
        text-align: center;
    }

    /* Input textarea */
    .stTextArea textarea {
        background-color: #374151;
        color: #f9fafb;
        border-radius: 15px;
        padding: 15px;
        font-size: 16px;
    }

    /* Button */
    .stButton>button {
        background-color: #f59e0b;
        color: #1f2937;
        font-size: 18px;
        height: 50px;
        width: 100%;
        border-radius: 15px;
        border: none;
        font-weight: bold;
    }

    /* Summary box */
    .summary-box {
        background-color: #111827;
        color: #f9fafb;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
        font-size: 16px;
        line-height: 1.6;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #374151;
        color: #f9fafb;
        border-radius: 15px;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Title and Description
# ==========================
st.title("📝 AI Text Summarizer")
st.markdown(
    "Paste any paragraph below, and our AI will generate a concise summary powered by **BART Large CNN**.",
    unsafe_allow_html=True
)

# ==========================
# Load summarization model
# ==========================
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# ==========================
# Input
# ==========================
text_input = st.text_area("Enter text here:", height=250)

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
        st.markdown(f"<div class='summary-box'>{summary[0]['summary_text']}</div>", unsafe_allow_html=True)

# ==========================
# Sidebar
# ==========================
st.sidebar.header("About")
st.sidebar.info(
    """
    - Built with **Streamlit** and **Hugging Face Transformers**  
    - Model: **facebook/bart-large-cnn**  
    - Dark theme for better readability  
    """
)
