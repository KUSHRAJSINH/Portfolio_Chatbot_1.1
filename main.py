import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#from langchain_core.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

# 🚨 CONFIGURATION
PDF_FILE_PATH = os.path.join("data", "Kushrajsinh_Zala_Resume_2025_Final_1310.pdf")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
# We need a dictionary to store conversation history per session/user (important for memory)
SESSION_MEMORY = {} 

# --- FastAPI Setup ---
app = FastAPI(title="RAG Portfolio API")

# 🚨 CRITICAL: The React app's URL MUST be in this list.
# If your React app is running at http://localhost:5173, include it!
origins = [
    "http://localhost:5173",    # <-- CHECK THIS LINE (Use the port shown in your terminal)
    "http://127.0.0.1:5173",    # <-- Good idea to include this too
    # Add your deployed Vercel domain here later: "https://kushraj-portfolio.vercel.app", 
    
    # You can use "*" for now to test, but specify domains for deployment
    # "*",  # Uncomment this ONLY for temporary testing if the specific ports fail
]
# 🚨 Add CORS Middleware: This allows your React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be specific with your domains later, but "*" works for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the request body from the frontend
class Query(BaseModel):
    # We'll use a session_id to maintain memory across user questions
    question: str
    session_id: str # A unique ID provided by the frontend

# --- RAG Chain Initialization (Runs once when the server starts) ---

def create_rag_chain():
    """Initializes the FAISS vector store and the prompt template."""
    
    # 1. Load and Split Document
    if not os.path.exists(PDF_FILE_PATH):
        raise FileNotFoundError(f"PDF file not found at: {PDF_FILE_PATH}")
    
    loader = PyPDFLoader(PDF_FILE_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    # 2. Setup Embeddings and Vectorstore
   
    embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
                    )

    '''embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )'''
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    
    # 3. Define LLM and Prompt Template (using your exact prompt)
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        groq_api_key=GROQ_API_KEY,
        streaming=False # Must be False for standard API
    )

    system_prompt = """You are Portfolio AI Assistant for Kushrajsinh Zala.
Your goal is to provide **brief, professional, and low-token answers** based ONLY on the documents.

Guidelines:
- **Maximum Length:** Respond using a maximum of **3 sentences total** per turn, unless providing a bulleted list.
- **Brevity Priority:** Prioritize concise language to save tokens.
- **Formatting:** Always use clean markdown bullet points (`* `) for lists (projects, skills, achievements).

Projects:
- If asked about projects, you MUST follow these two steps exactly:
    1. **ONLY list the short titles of the projects.** Do NOT generate descriptions or summaries for them.
    2. **Crucially, end your response with this instruction:** "For detailed information on any project, please view the Portfolio section of Kushraj's website, then ask me about that specific project by name."

Work Experience / Education / Skills:
- Provide answers using simple, short bullet points. Do NOT write full paragraphs.

Other:
- If asked something outside the portfolio's context, politely state that the answer is not available in the provided documents.
"""    
    # NOTE: Using the exact prompt structure from your Streamlit code
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

# Initialize RAG components globally
try:
    llm, retriever, prompt_template = create_rag_chain()
except Exception as e:
    print(f"FATAL: Could not initialize RAG chain: {e}")
    llm = None # Mark as failed if initialization fails

# --- Utility to get or create memory for a session ---
def get_session_chain(session_id):
    if session_id not in SESSION_MEMORY:
        # Create new memory and chain for a new session/user
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

# --- API Endpoint ---
@app.post("/ask")
async def ask_question(query: Query):
    if not llm:
        raise HTTPException(status_code=500, detail="RAG service failed to initialize. Check PDF and API key.")
    
    # Get the specific chain instance for this user session
    rag_chain_session = get_session_chain(query.session_id)
    
    try:
        # Invoke the chain with the user's question
        response = rag_chain_session.invoke({"question": query.question})
        
        # Return the AI's answer
        return {"answer": response['answer']}
    except Exception as e:
        print(f"Error processing question for session {query.session_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating AI response.")

# --- Root Endpoint for Health Check (Optional but recommended) ---
@app.get("/")
def read_root():
    if llm:
        return {"status": "RAG API is online and initialized successfully."}
    else:
        raise HTTPException(status_code=500, detail="RAG API is running but failed initialization.")