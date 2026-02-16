# ===============================================================
# STREAMLIT RAG CHATBOT USING LANGCHAIN + OPENAI
# ===============================================================

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# -------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot with OpenAI & LangChain")

# -------------------------------
# OPENAI API KEY (HARDCODED)
# -------------------------------
OPENAI_KEY = "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"

# -------------------------------
# DOCUMENT UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload a text document (.txt)", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.success("📄 Document Loaded Successfully!")

    # -------------------------------
    # SPLIT TEXT INTO CHUNKS
    # -------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)
    st.info(f"📌 Document split into {len(chunks)} chunks.")

    # -------------------------------
    # CREATE EMBEDDINGS AND VECTOR DB
    # -------------------------------
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    vector_db = FAISS.from_texts(chunks, embeddings)
    st.success("✅ Vector Database Created")

    # -------------------------------
    # CREATE LLM
    # -------------------------------
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=OPENAI_KEY
    )
    st.success("✅ LLM Ready")

    # -------------------------------
    # PROMPT TEMPLATE
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
    # USER QUESTION INPUT
    # -------------------------------
    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        # Retrieve relevant document chunks
        docs = vector_db.similarity_search(user_question, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Create final prompt
        prompt = prompt_template.format(
            context=context,
            question=user_question
        )

        # Generate answer using ChatOpenAI
        response = llm([HumanMessage(content=prompt)])
        st.subheader("Answer:")
        st.write(response[0].content)
