import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="FREE Study Bot", layout="wide")
st.title("📘 FREE Study Assistant (PDF Reference Based)")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdf_text(pdf_file):
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "text": text,
                    "page": i + 1
                })
    return pages

def chunk_text(pages, chunk_size=800, overlap=100):
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({
                "text": text[start:end],
                "page": p["page"]
            })
            start = end - overlap
    return chunks

def build_faiss(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def ask_question(question, index, chunks, k=5):
    q_embed = embedder.encode([question])
    D, I = index.search(np.array(q_embed).astype("float32"), k)

    answers = []
    pages = set()
    for idx in I[0]:
        answers.append(chunks[idx]["text"])
        pages.add(chunks[idx]["page"])

    return "\n\n".join(answers), sorted(pages)

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

uploaded_pdf = st.file_uploader("Upload PDF (large PDFs supported)", type="pdf")

if uploaded_pdf and st.button("Process PDF"):
    with st.spinner("Indexing PDF..."):
        pages = extract_pdf_text(uploaded_pdf)
        chunks = chunk_text(pages)
        index, chunks = build_faiss(chunks)
        st.session_state.index = index
        st.session_state.chunks = chunks
    st.success("PDF indexed successfully")

if st.session_state.index:
    question = st.text_input("Ask your question from the PDF")
    if question:
        answer, ref_pages = ask_question(
            question,
            st.session_state.index,
            st.session_state.chunks
        )
        st.markdown("### Answer (from document)")
        st.write(answer)
        st.markdown("### Reference Pages")
        st.write(ref_pages)


