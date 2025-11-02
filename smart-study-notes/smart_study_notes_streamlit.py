import streamlit as st
import os
import io
import textwrap
from typing import List

# Optional PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

# New OpenAI client import (v1.x syntax)
from openai import OpenAI

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PDFPLUMBER_AVAILABLE:
        try:
            text_pages = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for p in pdf.pages:
                    text_pages.append(p.extract_text() or "")
            return "\n\n".join(text_pages)
        except Exception:
            pass

    if PYPDF2_AVAILABLE:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            texts = []
            for p in reader.pages:
                try:
                    texts.append(p.extract_text() or "")
                except Exception:
                    texts.append("")
            return "\n\n".join(texts)
        except Exception:
            pass

    raise RuntimeError("No working PDF extractor found. Install pdfplumber or PyPDF2.")


def read_text_file(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except Exception:
        return file_bytes.decode("latin-1")


def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0
    for p in paragraphs:
        if not p.strip():
            continue
        if current_len + len(p) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += len(p) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks

# ---------------------------------------------------------------------------
# OpenAI helpers (new SDK)
# ---------------------------------------------------------------------------

def get_client(api_key: str = None):
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or input in sidebar.")
    return OpenAI(api_key=key)


def summarize_chunks(chunks: List[str], model: str, client: OpenAI) -> str:
    summaries = []
    for c in chunks:
        prompt = f"You are an assistant that converts lecture text into concise study bullet points.\n\nText:\n{c}\n\nProduce a short summary of the main points in 6-12 bullet points." 
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        summaries.append(resp.choices[0].message.content.strip())

    combined = "\n\n".join(summaries)
    prompt2 = f"Combine and deduplicate the following summarized bullet points into a single coherent set of 10-20 bullets suitable for study notes.\n\n{combined}"
    resp2 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt2}],
        max_tokens=800,
        temperature=0.2,
    )
    return resp2.choices[0].message.content.strip()


def generate_key_terms(text: str, model: str, client: OpenAI) -> List[str]:
    prompt = ("From the following lecture text, extract 12–20 important key terms or concepts. "
              "Return as a comma-separated list.\n\nText:\n" + text)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
    )
    out = resp.choices[0].message.content.strip()
    items = [i.strip() for i in out.replace("\n", ",").split(",") if i.strip()]
    return items[:20]


def generate_quiz(text: str, count: int, model: str, client: OpenAI) -> str:
    prompt = (f"Create {count} short quiz questions (mix of multiple-choice and short-answer) from the lecture text. "
              "For multiple choice provide 4 options and mark the correct answer.\n\nText:\n" + text)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Smart Study Notes Generator", layout="wide")
    st.title("🧠 Smart Study Notes Generator")
    st.markdown("Upload a lecture PDF or .txt and get concise study notes, key terms, and quiz questions.")

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key (or set OPENAI_API_KEY env var)", type="password")
        model = st.selectbox("Model", options=["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-2024"], index=0)
        quiz_count = st.slider("Number of quiz questions", 5, 10, 7)

    uploaded = st.file_uploader("Upload .pdf or .txt", type=["pdf", "txt"], accept_multiple_files=False)

    if uploaded is not None:
        raw = uploaded.read()
        try:
            if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(raw)
            else:
                text = read_text_file(raw)
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            return

        st.success(f"Text extracted — length: {len(text)} characters")

        with st.expander("Preview (first 1000 chars)"):
            st.text(textwrap.shorten(text, width=1000, placeholder="..."))

        if st.button("Generate Study Notes"):
            try:
                client = get_client(api_key)
                chunks = chunk_text(text, max_chars=3800)
                with st.spinner(f"Summarizing {len(chunks)} chunk(s)..."):
                    final_summary = summarize_chunks(chunks, model, client)

                st.header("Study Notes")
                st.markdown(final_summary)

                with st.spinner("Extracting key terms..."):
                    key_terms = generate_key_terms(text, model, client)
                st.subheader("Key Terms")
                st.write(key_terms)

                with st.spinner("Generating quiz questions..."):
                    quiz = generate_quiz(text, quiz_count, model, client)
                st.subheader("Quiz Questions")
                st.markdown(quiz)

                summary_txt = (f"Study Notes:\n\n{final_summary}\n\nKey Terms:\n\n{', '.join(key_terms)}\n\nQuiz:\n\n{quiz}")
                st.download_button("Download summary (.txt)", data=summary_txt, file_name="study_notes.txt")

            except Exception as e:
                st.error(f"Failed to generate notes: {e}")

    else:
        st.info("Upload a PDF or TXT file to begin.")

    st.markdown("---")
    st.caption("Built with Streamlit, pdfplumber / PyPDF2, and OpenAI's new Python SDK.")

if __name__ == "__main__":
    main()
