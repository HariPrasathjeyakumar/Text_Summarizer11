import streamlit as st
from transformers import pipeline

st.title("📝 AI Text Summarizer")

# Load model (will download from Hugging Face if not cached)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text_input = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if len(text_input.strip()) == 0:
        st.warning("Please enter some text!")
    else:
        input_words = len(text_input.split())
        max_len = min(250, max(120, int(input_words * 0.9)))
        min_len = max(80, int(max_len * 0.5))
        summary = summarizer(text_input, max_length=max_len, min_length=min_len, do_sample=False)
        st.subheader("✅ Summary")
        st.write(summary[0]['summary_text'])
