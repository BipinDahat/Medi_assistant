from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# STEP 1: Load a single PDF
# -------------------------------
PDF_FILE = r"C:\Medi_assistant\data\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"

def load_pdf_file(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

documents = load_pdf_file(PDF_FILE)
print(f"Loaded {len(documents)} pages from PDF.")


# -------------------------------
# STEP 2: Create Chunks
# -------------------------------
def create_chunks(text_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(text_data)
    return chunks

text_chunks = create_chunks(documents)
print(f"Created {len(text_chunks)} text chunks.")


# -------------------------------
# STEP 3: Embedding Model
# -------------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = get_embedding_model()


# -------------------------------
# STEP 4: Store in FAISS
# -------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print(f"FAISS Vector Database saved at: {DB_FAISS_PATH}")
