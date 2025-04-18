import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate

# --- Environment Fixes ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_NO_PYTHON_EXTENSION_WARNING"] = "1"

# --- Load and prepare data ---
st.title("🧠 FAISS RAG Dashboard (Search + Q&A + t-SNE)")

# Load data
loader = TextLoader("data.txt")
documents = loader.load()
cleaned_docs = [doc.page_content.strip() for doc in documents]

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separators=["\n\n", "\n", ".", " ", ""]
)
docs = splitter.create_documents(cleaned_docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# Dense retriever (FAISS)
dense_retriever = db.as_retriever()

# Sparse retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.3, 0.7]  # You can tweak this
)

# --- LLM for Q&A ---
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    #model="google/flan-t5-large"
    max_length=256,
    temperature=0.2,
)
llm = HuggingFacePipeline(pipeline=qa_pipeline)
prompt_template = PromptTemplate.from_template(
    "Use the context below to answer the question in a clear, complete, and human-friendly way.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=hybrid_retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# --- UI Display ---
num_docs = len(docs)
st.markdown(f"**📄 Document chunks stored:** `{num_docs}`")

# Show chunks
doc_list = [{"ID": i+1, "Content": doc.page_content[:200] + "..."} for i, doc in enumerate(docs)]
st.subheader("📘 Stored Chunks")
st.dataframe(pd.DataFrame(doc_list))

# --- Mode Switch ---
st.subheader("🧭 Choose Mode")
mode = st.radio("Select mode:", ["Semantic Search", "Q&A with LLM"])

query = st.text_input("Enter your query:")

if query:
    if mode == "Semantic Search":
        st.markdown("**🔍 Top Matching Chunks:**")
        results = db.similarity_search(query, k=5)
        for i, doc in enumerate(results):
            st.markdown(f"**Result {i+1}:** {doc.page_content}")
    elif mode == "Q&A with LLM":
        st.markdown("**🧠 Answer:**")
        result = qa_chain.invoke({"query": query})
        st.write(result["result"])

# --- Embedding Visualization ---
st.subheader("🧬 Embedding Visualization (t-SNE)")

if num_docs < 2:
    st.warning("You need at least 2 chunks for t-SNE.")
else:
    texts = [doc.page_content for doc in docs]
    emb_array = np.array(embeddings.embed_documents(texts))
    n_samples = emb_array.shape[0]
    safe_perplexity = min(30, n_samples - 1)

    tsne = TSNE(n_components=2, perplexity=safe_perplexity, random_state=42)
    emb_2d = tsne.fit_transform(emb_array)

    fig, ax = plt.subplots()
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.7)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE Visualization of Embeddings")
    st.pyplot(fig)
