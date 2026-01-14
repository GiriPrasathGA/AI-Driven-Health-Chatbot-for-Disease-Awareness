import os
from dotenv import load_dotenv
from typing import List, Tuple
import requests
from urllib.parse import quote_plus
import re
from PIL import Image
import pytesseract
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LangChain Imports ---
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# --- Translation ---
from deep_translator import GoogleTranslator

# --- Load Environment Variables ---
load_dotenv()

VECTOR_STORE_DIR = "FAISS/faiss_medicine_index_combined"
EMBED_MODEL = "embed-english-light-v3.0"
LLM_MODEL = "llama-3.1-8b-instant"
RETRIEVER_K = 4

# --- FastAPI Setup ---
app = FastAPI(title="AI Medicine Assistant (with WHO)", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings, vector_store, sessions = None, None, {}

# --- Translation Helpers ---
def detect_lang(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').detect(text)
    except Exception:
        return "en"

def translate_text(text: str, target_lang: str = "en") -> str:
    if not text:
        return text
    if target_lang.lower() == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        return text

# --- WHO API Fetcher ---
def fetch_who_info(disease_name: str):
    """Fetch related info from WHO API (health topics or ICD-11)."""
    try:
        q = quote_plus(disease_name)
        url = f"https://id.who.int/icd/release/11/2024/mms/search?q={q}"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            entities = data.get("destinationEntities") or []
            if entities:
                ent = entities[0]
                return {
                    "title": ent.get("title"),
                    "definition": ent.get("definition"),
                    "icd_url": ent.get("browserUrl")
                }
    except Exception:
        pass
    return None

# --- Startup: Load FAISS and Embeddings ---
@app.on_event("startup")
async def startup_event():
    global embeddings, vector_store
    print("🔹 Loading AI Medicine Assistant...")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model=EMBED_MODEL)
    vector_store = FAISS.load_local(
        VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    print("✅ FAISS index loaded successfully.")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    question: str
    chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str

# --- RAG Chain Setup ---
def create_rag_chain(vectorstore):
    llm = ChatGroq(
        temperature=0.5,
        model_name=LLM_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    MAIN_PROMPT = PromptTemplate(
        template="""
You are a compassionate AI Medicine Assistant.
Answer clearly, empathetically, and medically accurately using ONLY the context below.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={'k': RETRIEVER_K})
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, combine_docs_chain_kwargs={"prompt": MAIN_PROMPT}
    )

# --- Greeting Handler ---
def handle_greetings(user_input: str, lang: str) -> str:
    greetings = ["hi", "hello", "hey", "வணக்கம்"]
    if user_input.lower() in greetings:
        return translate_text(
            "Hello! I'm your AI Medicine Assistant. How can I help you today?",
            lang
        )
    return ""

# --- NEW: Image Upload Endpoint ---
@app.post("/upload_prescription", response_model=ChatResponse)
async def upload_prescription(
    session_id: str = Form(...),
    question: str = Form(""),
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))

        # Use OCR to extract text
        prescription_text = pytesseract.image_to_string(image)
        
        if not prescription_text.strip():
            return ChatResponse(answer="I couldn't read any text from that image. Please try a clearer picture.")

        llm = ChatGroq(
            temperature=0.5,
            model_name=LLM_MODEL,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        PRESCRIPTION_PROMPT = PromptTemplate(
            template="""
You are an AI assistant helping a user understand their medical prescription.
Analyze the following text, which was extracted from a doctor's prescription, and provide a clear, easy-to-understand summary.
Identify the medications, dosages, and instructions. Do NOT provide a diagnosis or treatment plan.
Always add a disclaimer to consult a doctor.

Prescription Text:
{text}

Question: {question}

Summary:
""",
            input_variables=["text", "question"]
        )

        chain = PRESCRIPTION_PROMPT | llm
        
        result = chain.invoke({
            "text": prescription_text,
            "question": question or "Summarize the prescription."
        })

        answer = result.content.strip()

        final_answer = answer + "\n\n⚕️ Please consult a healthcare professional for personalized advice."
        
        return ChatResponse(answer=final_answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# --- Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not vector_store:
        raise HTTPException(status_code=500, detail="Knowledge base not loaded")

    input_text = request.question.strip()
    if not input_text:
        raise HTTPException(status_code=400, detail="Empty question")

    user_lang = detect_lang(input_text) or "en"

    greeting_reply = handle_greetings(input_text, user_lang)
    if greeting_reply:
        return ChatResponse(answer=greeting_reply)

    translated_q = translate_text(input_text, "en")

    who_info = fetch_who_info(translated_q)
    external_data = ""
    if who_info:
        external_data = f"""
WHO ICD-11 Information:
Title: {who_info.get('title')}
Definition: {who_info.get('definition')}
More Info: {who_info.get('icd_url')}
"""

    if request.session_id not in sessions:
        sessions[request.session_id] = {"rag": create_rag_chain(vector_store)}
    rag_chain = sessions[request.session_id]["rag"]

    try:
        result = rag_chain.invoke({
            "question": f"{translated_q}\n\n{external_data}",
            "chat_history": request.chat_history
        })
        answer = result.get("answer", "").strip()

        lines = answer.split("\n")
        unique_lines = []
        for line in lines:
            if line.strip() and (len(unique_lines) == 0 or line.strip() != unique_lines[-1].strip()):
                unique_lines.append(line)
        answer = "\n".join(unique_lines)

        if user_lang.lower() in ["ta", "tamil"]:
            answer = re.sub(r'(நீங்கள்\s+காய்ச்சல்\s+இருக்கும்\s+போது,?\s*){2,}', r'\1', answer)

        if len(answer) > 2000:
            answer = answer[:2000] + "..."

        answer += "\n\n⚕️ Please consult a healthcare professional for personalized advice."

        final_answer = translate_text(answer, user_lang)

        return ChatResponse(answer=final_answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)