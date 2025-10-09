import os
from dotenv import load_dotenv
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LangChain Imports ---
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# --- Translation ---
from deep_translator import GoogleTranslator

# --- Load Environment Variables ---
load_dotenv()

VECTOR_STORE_DIR = "FAISS/faiss_medicine_index_combined"
EMBED_MODEL = "embed-english-light-v3.0"
LLM_MODEL = "llama-3.1-8b-instant"
RETRIEVER_K = 4

# --- FastAPI Setup ---
app = FastAPI(title="AI Medicine Assistant (Simple)", version="1.0")

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
    if target_lang.lower() == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        return text

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
        temperature=0.3,
        model_name=LLM_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    MAIN_PROMPT = PromptTemplate(
        template="""
You are a compassionate AI Medicine Assistant. Speak naturally and empathetically.
Use ONLY the information in the given context.

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

# --- Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not vector_store:
        raise HTTPException(status_code=500, detail="Knowledge base not loaded")

    input_text = request.question.strip()
    user_lang = detect_lang(input_text) or "en"

    # --- Check for greetings ---
    greeting_reply = handle_greetings(input_text, user_lang)
    if greeting_reply:
        return ChatResponse(answer=greeting_reply)

    # --- Translate user query to English ---
    translated_q = translate_text(input_text, "en")

    # --- Create or reuse RAG chain session ---
    if request.session_id not in sessions:
        sessions[request.session_id] = {"rag": create_rag_chain(vector_store)}
    rag_chain = sessions[request.session_id]["rag"]

    try:
        # --- Call RAG Chain for detailed response ---
        result = rag_chain.invoke({
            "question": translated_q,
            "chat_history": request.chat_history
        })
        answer = result["answer"].strip()

        # Add gentle medical disclaimer
        answer += "\n\n⚕️ Please consult a healthcare professional for personalized advice."

        # Translate back to user language
        final_answer = translate_text(answer, user_lang)
        return ChatResponse(answer=final_answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
