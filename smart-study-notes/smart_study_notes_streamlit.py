import streamlit as st
import pdfplumber
import os
import google.generativeai as genai

# ---------------------------------------------------------
# 🧠 SMART STUDY NOTES GENERATOR (Gemini Free API)
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
    st.info("Create one at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)")
    st.stop()

# Configure Gemini client
genai.configure(api_key=api_key)

# ---------------------------------------------------------
# 📂 Upload section
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
    st.text_area("📜 Preview (first 1000 chars)", text[:1000], height=200)

    # ---------------------------------------------------------
    # 🚀 Generate Notes
    # ---------------------------------------------------------
    if st.button("✨ Generate Study Notes and Quiz"):
        with st.spinner("Generating notes using Gemini... ⏳"):
            try:
                prompt = f"""
                You are a helpful study assistant.
                Summarize the following text into concise, bullet-point study notes.
                Then create 5–10 quiz questions to test understanding of the material.

                Text:
                {text[:12000]}
                """

                # Supported model
                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                response = model.generate_content(prompt)

                result = response.text.strip()

                st.subheader("📘 Study Notes & Quiz")
                st.write(result)

                st.download_button(
                    label="📥 Download Summary",
                    data=result,
                    file_name="study_notes.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ Failed to generate notes: {e}")
else:
    st.info("⬆️ Please upload a .pdf or .txt file to begin.")
