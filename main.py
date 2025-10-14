import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# --- CHANGE START ---
# The deprecated import:
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# The recommended, updated import:
from langchain_huggingface import HuggingFaceEmbeddings # This is the key fix
# --- CHANGE END ---
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIG ---
PDF_FILE_PATH = os.path.join("data", "Kushrajsinh_Zala_Resume_2025_Final_1310.pdf")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SESSION_MEMORY = {}  # Session-based memory

# --- FastAPI Setup ---
app = FastAPI(title="RAG Portfolio API")

# --- CORS Configuration ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # Replace with your frontend domain after deployment
    # "https://your-portfolio.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Specific domains only
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
    if not os.path.exists(PDF_FILE_PATH):
        raise FileNotFoundError(f"PDF not found at: {PDF_FILE_PATH}")

    # Load and split PDF
    loader = PyPDFLoader(PDF_FILE_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    # Embeddings and vectorstore
    # --- CHANGE START ---
    # Using the updated class from langchain_huggingface
    embeddings = HuggingFaceEmbeddings(
        #model_name="BAAI/bge-small-en",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    # --- CHANGE END ---
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

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
        raise HTTPException(status_code=500, detail="RAG failed to initialize. Check PDF/API key.")

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
        return {"status": "RAG API online and initialized successfully."}
    else:
        raise HTTPException(status_code=500, detail="RAG API running but failed initialization.")