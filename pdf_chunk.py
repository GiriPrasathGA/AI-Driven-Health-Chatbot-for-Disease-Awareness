import os
import time
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------- Config -----------------
PDF_FILE = "part4.pdf"
VECTOR_STORE_DIR = "faiss_part4"
BATCH_SIZE = 80       # chunks per embedding batch
WAIT_TIME = 30        # wait if rate-limited (in seconds)

# ----------------- Load API key -----------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Please set COHERE_API_KEY in your .env file")

# ----------------- Read PDF -----------------
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n\n--- Page {i+1} ---\n\n" + page_text
    return text

print(f"📄 Reading PDF: {PDF_FILE}")
pdf_text = read_pdf(PDF_FILE)
print(f"✅ Extracted {len(pdf_text)} characters from PDF")

# ----------------- Split into Chunks -----------------
def split_pdf_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    texts = splitter.split_text(text)
    chunks = [{"title": f"chunk_{i}", "text": t} for i, t in enumerate(texts)]
    return chunks

chunks_with_titles = split_pdf_into_chunks(pdf_text)
print(f"✅ Split PDF into {len(chunks_with_titles)} chunks")

# ----------------- Initialize Cohere Embeddings -----------------
print("💡 Initializing Cohere embeddings...")
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-light-v3.0")

# ----------------- Create FAISS Vector Store -----------------
all_vectors = None
for i in range(0, len(chunks_with_titles), BATCH_SIZE):
    batch = chunks_with_titles[i:i + BATCH_SIZE]
    texts = [c["text"] for c in batch]
    metadatas = [{"title": c["title"]} for c in batch]
    
    try:
        print(f"🧩 Processing batch {i // BATCH_SIZE + 1}/{(len(chunks_with_titles) // BATCH_SIZE) + 1} ({len(texts)} chunks)")
        vs = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
        if all_vectors is None:
            all_vectors = vs
        else:
            all_vectors.merge_from(vs)
        time.sleep(2)  # prevent rate limits
    except Exception as e:
        print(f"⚠️ Rate limit or error: {e}. Waiting {WAIT_TIME} sec before retrying...")
        time.sleep(WAIT_TIME)
        continue

# ----------------- Save FAISS -----------------
all_vectors.save_local(VECTOR_STORE_DIR)
print(f"✅ FAISS vector store saved to '{VECTOR_STORE_DIR}'")