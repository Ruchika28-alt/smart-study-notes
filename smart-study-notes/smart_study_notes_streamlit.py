import streamlit as st
import pdfplumber
import os
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 🧠 SMART STUDY NOTES GENERATOR (Gemini Free Version)
# ---------------------------------------------------------

st.set_page_config(page_title="🧠 Smart Study Notes Generator", layout="wide")

st.title("🧠 Smart Study Notes Generator")
st.write("Upload lecture notes or PDFs → get concise study notes, key terms, and quiz questions.")

# ---------------------------------------------------------
# 🔑 Load Gemini API key
# ---------------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ Please add your Gemini API key in Streamlit Secrets or environment variable.")
    st.info("Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) → Create key → Paste it under 'Settings → Secrets' as GEMINI_API_KEY.")
    st.stop()

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# ---------------------------------------------------------
# 📂 File Upload Section
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📄 Upload your lecture notes (.pdf or .txt)", type=["pdf", "txt"])

if uploaded_file:
    # Extract text
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
    else:
        text = uploaded_file.read().decode("utf-8")

    if not text.strip():
        st.warning("⚠️ No readable text found in the file.")
        st.stop()

    st.success("✅ Text extracted successfully!")
    st.text_area("📜 Extracted Text Preview (first 1000 chars)", text[:1000], height=200)

    # ---------------------------------------------------------
    # 🚀 Generate Study Notes
    # ---------------------------------------------------------
    if st.button("✨ Generate Study Notes and Quiz"):
        with st.spinner("Generating notes using Gemini... ⏳"):
            try:
                prompt = f"""
                You are a smart study assistant. Summarize the following text into concise, easy-to-read study notes in bullet points.
                Also, create 5–10 quiz questions to test understanding of the content.

                Text:
                {text[:12000]}  # limit to prevent overflow
                """

                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.4,
                    ),
                )

                summary_output = response.text.strip()

                # Display Output
                st.subheader("📘 Generated Study Notes & Quiz")
                st.write(summary_output)

                # Download Option
                st.download_button(
                    label="📥 Download Summary as Text File",
                    data=summary_output,
                    file_name="study_notes.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ Failed to generate notes: {e}")
else:
    st.info("⬆️ Please upload a .pdf or .txt file to begin.")
