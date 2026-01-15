import streamlit as st
from PyPDF2 import PdfReader
import nltk

import nltk
import os
from pathlib import Path

NLTK_DIR = Path(os.getcwd()) / "nltk_data"
NLTK_DIR.mkdir(exist_ok=True)

nltk.data.path.append(str(NLTK_DIR))

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=str(NLTK_DIR))

st.set_page_config(page_title="Text Chunker (NLTK Sentence Tokenizer)", layout="wide")

st.title("Text Chunker (NLTK Sentence Tokenizer)")
st.write(
    "This app reads a **PDF**, extracts its text, splits it into **sentences** using **NLTK**, "
    "shows a sample of sentences (indices **58 to 68**), and performs **semantic sentence chunking**."
)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def extract_text_from_pdf(pdf_file) -> str:
    """Step 1 & 2: Import PDF using PdfReader and extract text."""
    reader = PdfReader(pdf_file)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)

def sentence_chunker(sentences, N: int):
    """Step 4: Chunk by number of sentences (semantic sentence chunking)."""
    chunks = []
    for i in range(0, len(sentences), N):
        chunks.append(" ".join(sentences[i:i+N]))
    return chunks

# -----------------------------
# UI
# -----------------------------
st.subheader("Step 1: Upload PDF")
pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

chunk_size = st.number_input(
    "Number of sentences per chunk (N)",
    min_value=1,
    max_value=200,
    value=5,
    step=1,
)

if st.button("Extract + Chunk"):
    if pdf is None:
        st.warning("Please upload a PDF file first.")
    else:
  
        extracted_text = extract_text_from_pdf(pdf)

        if not extracted_text.strip():
            st.error("No text could be extracted from this PDF (it may be scanned images).")
        else:
            st.success("PDF text extracted successfully.")

            sentences = nltk.sent_tokenize(extracted_text)

            st.write(f"Total sentences extracted: **{len(sentences)}**")

            st.subheader("Step 3: Sample extracted sentences (indices 58 to 68)")
            start_i, end_i = 58, 68

            if len(sentences) <= start_i:
                st.info(f"Not enough sentences to display indices {start_i}–{end_i}.")
            else:
                sample = sentences[start_i:min(end_i + 1, len(sentences))]
                for idx, s in enumerate(sample, start=start_i):
                    st.markdown(f"**[{idx}]** {s}")

            st.subheader("Step 4: Semantic sentence chunking (NLTK sentence tokenizer)")
            chunks = sentence_chunker(sentences, int(chunk_size))
            st.success(f"Number of sentence chunks = {len(chunks)}")

            idx = st.number_input(
                "Select chunk index to view",
                min_value=1,
                max_value=len(chunks),
                value=1,
                step=1,
            )
            st.subheader(f"Chunk {idx}")
            st.write(chunks[idx - 1])

            with st.expander("Show all chunks"):
                for i, ch in enumerate(chunks, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(ch)
                    st.markdown("---")
