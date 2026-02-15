# ==========================
# Streamlit RAG Chatbot App
# ==========================

import streamlit as st
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot with OpenAI Embeddings")

# -------------------------------
# OpenAI API Key from Streamlit Secrets
# -------------------------------
OPENAI_KEY = st.secrets["openai"]["api_key"]

# -------------------------------
# Upload Document
# -------------------------------
uploaded_file = st.file_uploader("Upload your text document", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    st.success("✅ Document Loaded")

    # -------------------------------
    # Split Text into Chunks
    # -------------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    st.info(f"Document split into {len(chunks)} chunks")

    # -------------------------------
    # Create Embeddings
    # -------------------------------
    embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
    vector_db = FAISS.from_texts(chunks, embeddings)
    st.success("✅ Vector Database Created")

    # -------------------------------
    # LLM Model
    # -------------------------------
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, api_key=OPENAI_KEY)

    # -------------------------------
    # Prompt Template
    # -------------------------------
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer using only the context below.
If answer not present, say "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    # -------------------------------
    # User Question Input
    # -------------------------------
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        # Retrieve similar chunks
        docs = vector_db.similarity_search(user_question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = prompt_template.format(context=context, question=user_question)

        # Get response from LLM
        response = llm(prompt)
        st.markdown("### 🤖 Answer")
        st.write(response.content)

else:
    st.info("Please upload a text file to get started.")
