import os
import re
from dotenv import load_dotenv
from typing import List, Tuple, Optional

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

# --- Configuration Constants ---
VECTOR_STORE_DIR = "faiss_medicine_index_combined"
EMBED_MODEL = "embed-english-light-v3.0"
LLM_MODEL = "llama-3.1-8b-instant"
RETRIEVER_K = 4

# --- FastAPI App ---
app = FastAPI(
    title="AI Medicine Assistant",
    description="RAG chatbot for medicine and symptoms knowledge (multi-language support).",
    version="1.1.0"
)

# --- CORS ---
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:5500"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables ---
embeddings, vector_store, sessions = None, None, {}

# --- Helper: split large text into sections ---
def split_text_by_section(text: str):
    sections = re.split(r'\n(?=## )', text)
    return [sec.strip() for sec in sections if sec.strip()]

# --- Translate text safely ---
def translate_text(text: str, target_lang: str = "en") -> str:
    if target_lang.lower() == "en":
        return text  # no translation needed
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        return text  # fallback to original if translation fails

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    global embeddings, vector_store
    print("--- Starting AI Medicine Assistant ---")
    
    # Initialize Cohere embeddings
    cohere_api_key = os.getenv("COHERE_API_KEY")
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model=EMBED_MODEL)
    print("✅ Cohere embeddings initialized")

    # Load FAISS vector store
    try:
        vector_store = FAISS.load_local(
            VECTOR_STORE_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ FAISS vector store loaded from '{VECTOR_STORE_DIR}'")
    except Exception as e:
        print(f"❌ Failed to load FAISS vector store: {e}")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    question: str
    lang: Optional[str] = "en"  # target language, default English
    chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str

# --- Core RAG Chain Creation ---
def create_rag_chain(vectorstore):
    llm = ChatGroq(
        temperature=0.1, 
        model_name=LLM_MODEL, 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    condense_template = """
    Given the following conversation and a follow-up question, rephrase it as a standalone question.
    
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

    main_prompt_template = """
You are a precise AI assistant for medicine and symptoms.

Rules:
1. Use ONLY the information in the 'Context'.
2. If the context doesn't contain the answer, respond: "I'm sorry, that information is not available in my knowledge base."
3. Use HTML formatting:
   - <b>Bold</b> for key terms
   - <ul><li>...</li></ul> for bullet points
   - <ol><li>...</li></ol> for steps

Context: 
{context}

Question: {question}

Response: Provide a structured answer in HTML format strictly from context.
"""
    MAIN_PROMPT = PromptTemplate(
        template=main_prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={'k': RETRIEVER_K})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": MAIN_PROMPT},
    )

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    question_lower = request.question.lower().strip()

    # Greeting / Identity / Farewell patterns
    greeting_pattern = r"^\s*(hi|hii|hello|hey|heyy)\s*!*$"
    identity_pattern = r"who are you|what are you|what is your purpose|tell me about yourself"
    farewell_pattern = r"bye|goodbye|good bye|thanks|thank you|see you|ok then good bye"

    if re.fullmatch(greeting_pattern, question_lower):
        response_text = "Hello! I'm your AI Medicine Assistant. How can I help you?"
        return ChatResponse(answer=translate_text(response_text, request.lang))

    if re.search(identity_pattern, question_lower):
        response_text = "I am an AI assistant for medicine knowledge. I answer questions strictly based on my knowledge base."
        return ChatResponse(answer=translate_text(response_text, request.lang))

    if re.search(farewell_pattern, question_lower):
        response_text = "You're welcome! Goodbye and take care."
        return ChatResponse(answer=translate_text(response_text, request.lang))

    # Use RAG chain
    try:
        if not vector_store:
            response_text = "I'm sorry, my knowledge base is currently unavailable."
            return ChatResponse(answer=translate_text(response_text, request.lang))
        
        if request.session_id not in sessions:
            sessions[request.session_id] = {"rag_chain": create_rag_chain(vector_store)}
        
        rag_chain = sessions[request.session_id]["rag_chain"]
        result = rag_chain.invoke({
            "question": request.question,
            "chat_history": request.chat_history
        })

        translated_answer = translate_text(result["answer"].strip(), request.lang)
        return ChatResponse(answer=translated_answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"--- Starting server on http://0.0.0.0:{port} ---")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
