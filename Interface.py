import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

# ---------------------- SETUP ----------------------
st.set_page_config(
    page_title="Medi Assistant",
    page_icon="🩺",
    layout="wide"
)

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# ---------------------- SIMPLE CSS ----------------------
st.markdown("""
<style>
    body {
        background-color: #f7f9fc;
    }
    .msg-box {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .user {
        background-color: #dce8ff;
        color: #003366;
    }
    .assistant {
        background-color: #d9f7e8;
        color: #004d33;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.title("🩺 Medi Assistant")
st.write("Ask medical questions based on your uploaded documents.")

# ---------------------- SIDEBAR SETTINGS ----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
    max_tokens = st.slider("Max Tokens", 256, 2048, 512)
    k_value = st.slider("Retrieve Top-k Documents", 1, 10, 3)

st.divider()

# ---------------------- LOAD VECTORSTORE ----------------------
vectorstore = load_vectorstore()

# ---------------------- CHAT AREA ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    css_class = "user" if msg["role"] == "user" else "assistant"
    st.markdown(f"<div class='msg-box {css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# Input box
user_prompt = st.chat_input("Type your question...")

# ---------------------- PROCESS USER QUERY ----------------------
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.markdown(f"<div class='msg-box user'>{user_prompt}</div>", unsafe_allow_html=True)

    try:
        # LLM Setup
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.environ.get("GROQ_API_KEY")
        )

        # RAG chain
        retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_chain = create_stuff_documents_chain(llm, retrieval_prompt)

        rag_chain = create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={'k': k_value}),
            combine_chain
        )

        # Generate answer
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke({"input": user_prompt})
            answer = response.get("answer", "No response generated.")

        st.markdown(f"<div class='msg-box assistant'>{answer}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Error: {str(e)}")

# ---------------------- FOOTER ----------------------
st.divider()
st.caption("© 2025 Medi Assistant – Powered by LangChain, Groq & HuggingFace")
