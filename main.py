

# ---------------------------------------------------------
# IMPORT STREAMLIT (UI FRAMEWORK)
# Used to build the web interface (buttons, input, file upload)
# ---------------------------------------------------------
import streamlit as st

# ---------------------------------------------------------
# IMPORT PyMuPDF (PDF PROCESSING LIBRARY)
# Used to read PDF files and extract text + render pages
# ---------------------------------------------------------
import fitz

# ---------------------------------------------------------
# OS MODULE
# Used for file system operations (like saving images)
# ---------------------------------------------------------
import os

# ---------------------------------------------------------
# NUMPY
# Used for handling numerical arrays (embeddings)
# ---------------------------------------------------------
import numpy as np

# ---------------------------------------------------------
# FAISS
# Fast similarity search library for vector retrieval
# ---------------------------------------------------------
import faiss

# ---------------------------------------------------------
# PIL (Python Imaging Library)
# Used to open and display images in Streamlit
# ---------------------------------------------------------
from PIL import Image

# ---------------------------------------------------------
# OPENAI CLIENT (USED WITH OPENROUTER)
# Used to call LLM APIs for generating answers
# ---------------------------------------------------------
from openai import OpenAI

# ---------------------------------------------------------
# SENTENCE TRANSFORMERS
# Converts text into embeddings (vector representations)
# ---------------------------------------------------------
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# LANGCHAIN TEXT SPLITTER
# Splits large text into smaller chunks for better retrieval
# ---------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# OPENROUTER API SETUP
# This connects your app to an external LLM (GPT-like model)
# ---------------------------------------------------------
client = OpenAI(
    api_key="OPENROUTER_API_KEY",
    base_url="https://openrouter.ai/api/v1"
)


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL (CACHED FOR PERFORMANCE)
# Caching avoids reloading model every time app runs
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Load lightweight transformer model for embeddings
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# INITIALIZE MODEL
# Embedding model used throughout the app
# ---------------------------------------------------------
model = load_model()


# ---------------------------------------------------------
# FUNCTION: PROCESS PDF
# Converts PDF → text → chunks → embeddings → FAISS index
# ---------------------------------------------------------
def process_pdf(pdf_file):

    # Open uploaded PDF from memory stream
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")

    # Store extracted text page by page
    documents = []

    # Loop through all pages in PDF
    for page_num, page in enumerate(pdf):

        # Extract text from current page
        text = page.get_text()

        # Store text + page number
        documents.append({
            "text": text,
            "page": page_num + 1
        })

    # Create text splitter for chunking large text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # max size of each chunk
        chunk_overlap=50     # overlap helps preserve context
    )

    # Store final chunked data
    chunked_documents = []

    # Loop through each page document
    for doc in documents:

        # Split page text into smaller chunks
        chunks = splitter.split_text(doc["text"])

        # Store each chunk separately with page reference
        for chunk in chunks:
            chunked_documents.append({
                "text": chunk,
                "page": doc["page"]
            })

    # Extract only text for embedding generation
    texts = [c["text"] for c in chunked_documents]

    # Convert text into vector embeddings
    embeddings = model.encode(texts)

    # Convert embeddings to float32 (required by FAISS)
    embeddings = np.array(embeddings).astype("float32")

    # Get vector dimension size
    dim = embeddings.shape[1]

    # Create FAISS index for similarity search (L2 distance)
    index = faiss.IndexFlatL2(dim)

    # Add embeddings into FAISS index
    index.add(embeddings)

    # Return processed chunks + search index
    return chunked_documents, index


# ---------------------------------------------------------
# STREAMLIT UI HEADER
# Displays title of the web app
# ---------------------------------------------------------
st.title("📄PDF-AI-Bot")

# ---------------------------------------------------------
# UI DESCRIPTION TEXT
# Explains what the app does to user
# ---------------------------------------------------------
st.write("Upload your PDF and ask questions from it")


# ---------------------------------------------------------
# FILE UPLOADER
# Lets user upload PDF file
# ---------------------------------------------------------
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])


# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# Used to store processed PDF data across interactions
# ---------------------------------------------------------
if "data" not in st.session_state:
    st.session_state.data = None


# ---------------------------------------------------------
# PROCESS PDF BUTTON
# Runs full PDF → embedding pipeline
# ---------------------------------------------------------
if pdf_file and st.button("Process PDF"):

    # Show loading animation while processing
    with st.spinner("Processing PDF..."):

        # Process PDF into chunks + FAISS index
        chunked_docs, index = process_pdf(pdf_file)

        # Store results in session memory
        st.session_state.data = {
            "chunks": chunked_docs,
            "index": index
        }

    # Show success message
    st.success("PDF processed successfully!")


# ---------------------------------------------------------
# USER QUESTION INPUT
# Text box where user asks question
# ---------------------------------------------------------
question = st.text_input("Ask your question")


# ---------------------------------------------------------
# GET ANSWER BUTTON
# Runs retrieval + LLM response generation
# ---------------------------------------------------------
if st.button("Get Answer"):

    # Check if PDF has been processed
    if st.session_state.data is None:
        st.error("Please upload and process a PDF first")
        st.stop()

    # Load stored chunks and index
    chunks = st.session_state.data["chunks"]
    index = st.session_state.data["index"]

    # Convert question into embedding vector
    q_emb = model.encode([question])

    # Convert to FAISS-compatible format
    q_emb = np.array(q_emb).astype("float32")

    # Search top 3 most similar chunks
    distances, indices = index.search(q_emb, 3)

    # Retrieve actual text chunks
    retrieved = [chunks[i] for i in indices[0]]

    # Combine retrieved chunks into context
    context = "\n\n".join([r["text"] for r in retrieved])


    # -----------------------------------------------------
    # CALL LLM (OPENROUTER)
    # Generates answer using retrieved context
    # -----------------------------------------------------
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY using the provided PDF context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )

    # Show AI response header
    st.subheader("🧠 Answer")

    # Display generated answer
    st.write(response.choices[0].message.content)


    # ---------------------------------------------------------
    # REFERENCE PAGE DISPLAY SECTION
    # Shows PDF pages used for answering
    # ---------------------------------------------------------
    st.subheader("📌 Reference Pages")

    # Create folder for storing rendered page images
    os.makedirs("reference_pages", exist_ok=True)

    # Track already shown pages (avoid duplicates)
    shown = set()

    # Loop through retrieved chunks
    for r in retrieved:

        # Extract page number from chunk
        page_num = r["page"]

        # Skip if already shown
        if page_num in shown:
            continue

        # Mark page as shown
        shown.add(page_num)

        # Define image path for saving page render
        img_path = f"reference_pages/page_{page_num}.png"

        # Reopen PDF for rendering page
        pdf = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")

        # Get specific page
        page = pdf[page_num - 1]

        # Convert page to image
        pix = page.get_pixmap()

        # Save image file
        pix.save(img_path)

        # Display image in Streamlit UI
        st.image(Image.open(img_path), caption=f"Page {page_num}")
