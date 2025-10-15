import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIG ---
FAISS_INDEX_PATH = "faiss_index" 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SESSION_MEMORY = {}

# --- FastAPI Setup ---
app = FastAPI(title="RAG Portfolio API")

# --- CORS Configuration ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.0.1:5173",
    # "https://your-portfolio.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic model ---
class Query(BaseModel):
    question: str
    session_id: str

# --- RAG Chain Initialization ---
def create_rag_chain():
    """Initialize FAISS vector store, embeddings, LLM, and prompt template."""
    
    if not os.path.exists(FAISS_INDEX_PATH) or not os.listdir(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FATAL: Pre-built FAISS index not found at: {FAISS_INDEX_PATH}. The 'faiss_index' folder must be present.")

    # Embeddings and vectorstore
    # --- UPDATED MODEL NAME ---
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", # Changed to BGE-small for better semantic retrieval
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # --- UPDATED RETRIEVER SETTINGS ---
    # Retrieve 6 chunks instead of the default (usually 4) to ensure more context is included
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6}) 

    # LLM and prompt template
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY,
        streaming=False
    )
    system_prompt = """
    You are Portfolio AI Assistant for Kushrajsinh Zala.  
    - Answer briefly (max 3 sentences).  
    - Use bullets for lists.  
    - Projects: list titles only; if asked for details, reply: "Please visit the Portfolio section and provide the project name for details."  
    - Work Experience: always show Petpooja first, then Zidio Development.  
    - Education: show University first, then School.  
    - If info not found in documents or vector DB, reply: "I don't know."  
    - Questions outside portfolio: reply politely, "Answer not available."  
    """

    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=f"""{system_prompt}
Chat history:
{{chat_history}}
Relevant context from documents:
{{context}}
User question: {{question}}
Your response:"""
    )

    return llm, retriever, prompt_template

# Initialize globally
try:
    llm, retriever, prompt_template = create_rag_chain()
except Exception as e:
    print(f"FATAL: Could not initialize RAG chain: {e}")
    llm = None

# --- Session-based chain ---
def get_session_chain(session_id):
    if session_id not in SESSION_MEMORY:
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )
        SESSION_MEMORY[session_id] = chain
    return SESSION_MEMORY[session_id]

# --- API Endpoints ---
@app.post("/ask")
async def ask_question(query: Query):
    if not llm:
        raise HTTPException(status_code=500, detail="RAG failed to initialize. Check FAISS index/API key.")

    chain = get_session_chain(query.session_id)
    try:
        response = chain.invoke({"question": query.question})
        return {"answer": response['answer']}
    except Exception as e:
        print(f"Error for session {query.session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating AI response.")

# Health check
@app.get("/")
def read_root():
    if llm:
        return {"status": "RAG API online and initialized successfully (Loaded pre-built index)."}
    else:
        raise HTTPException(status_code=500, detail="RAG API running but failed initialization. Check the 'faiss_index' folder and GROQ_API_KEY.")