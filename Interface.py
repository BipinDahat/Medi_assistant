import os
import streamlit as st
from streamlit.components.v1 import html
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_core.prompts import PromptTemplate

# ---------------------- CONFIGURATION ----------------------
st.set_page_config(
    page_title="SympAI - Smart Medical Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# ---------------------- CUSTOM STYLES ----------------------
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: #ffffff;
        }
        .stChatMessage {
            background-color: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 10px;
        }
        .user-msg {
            background-color: #1f4e79;
            padding: 10px;
            border-radius: 12px;
            color: white;
        }
        .assistant-msg {
            background-color: #2e7d32;
            padding: 10px;
            border-radius: 12px;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #1b263b;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- VECTOR STORE ----------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# ---------------------- CUSTOM PROMPT ----------------------
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# ---------------------- HEADER ----------------------
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=90)
with col2:
    st.markdown("<h1 style='color:#00e6e6; font-size: 42px;'>🧬 SympAI: Intelligent Health Q&A System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#cceeff;'>Ask questions from medical or healthcare documents and get accurate AI-powered insights!</p>", unsafe_allow_html=True)

st.divider()

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966487.png", width=120)
    st.markdown("### ⚙️ Settings")
    st.write("Adjust chatbot parameters below:")

    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.5)
    max_tokens = st.slider("Max Tokens", 256, 2048, 512)
    k_val = st.slider("Number of Retrieved Docs (k)", 1, 10, 3)

    st.markdown("---")
    st.markdown("### 💡 About SympAI")
    st.info("SympAI uses Groq + HuggingFace embeddings + FAISS to retrieve answers from your local vector store.")

# ---------------------- MAIN CHAT AREA ----------------------
st.markdown("<h3 style='color:#80dfff;'>💬 Chat with SympAI</h3>", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    role_class = 'user-msg' if message['role'] == 'user' else 'assistant-msg'
    st.markdown(f"<div class='{role_class}'>{message['content']}</div>", unsafe_allow_html=True)

# ---------------------- USER INPUT ----------------------
prompt = st.chat_input("Type your question...")

if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.markdown(f"<div class='user-msg'>{prompt}</div>", unsafe_allow_html=True)

    with st.spinner("🔍 Retrieving the best answer... Please wait..."):
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store.")
            else:
                GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=GROQ_API_KEY,
                )

                retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
                combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
                rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': k_val}), combine_docs_chain)
                
                response = rag_chain.invoke({'input': prompt})
                result = response.get("answer", "No response generated.")
                
                st.markdown(f"<div class='assistant-msg'>{result}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# ---------------------- FOOTER ----------------------
st.divider()
st.markdown("""
    <p style='text-align: center; color: #cccccc; font-size: 13px;'>
    © 2025 SympAI | Powered by LangChain, Groq & HuggingFace
    </p>
""", unsafe_allow_html=True)
