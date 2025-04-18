import streamlit as st
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama


# --- Env Fix ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_NO_PYTHON_EXTENSION_WARNING"] = "1"

st.set_page_config(page_title="Local RAG with Ollama + Milvus", layout="wide")
st.title("🧠 Local RAG Q&A - MapReduce + Ollama + Milvus")

# Optional: for BM25 + visualization
import time

# Optional: for BM25 + visualization
@st.cache_resource(show_spinner=False)
def load_docs():
    st.write("📥 Loading and splitting documents...")
    start = time.time()
    
    loader = TextLoader("data.txt")
    raw_docs = loader.load()
    st.write(f"✅ Loaded raw documents. Count: {len(raw_docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    chunks = splitter.split_documents(raw_docs)
    
    end = time.time()
    st.write(f"🧩 Split into {len(chunks)} chunks in {end - start:.2f} seconds.")
    
    return chunks

# Only needed for BM25 and t-SNE
docs = load_docs()

# --- Connect to Milvus Vector Store ---
@st.cache_resource(show_spinner=False)
def connect_milvus():
    st.write("🔗 Connecting to Milvus and loading embeddings...")
    start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Milvus(
        embedding_function=embeddings,
        collection_name="langchain_docs",
        connection_args={"host": "localhost", "port": "19530"}
    )
    st.write(f"✅ Milvus connected and embeddings ready in {time.time() - start:.2f}s")
    return vector_db, embeddings

vector_db, embeddings = connect_milvus()

# --- Set Up Retrievers ---
st.write("🔍 Setting up retrievers...")
dense_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3
retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# --- Prompt and LLM Setup ---
prompt = PromptTemplate.from_template(
    """You are a helpful assistant.
Answer the question using only the context below.

Context:
{context}

Question: {question}

Helpful Answer:"""
)

llm = ChatOllama(
    model="llama3",
    temperature=0.7,
    
)   # Requires `ollama serve` running with llama3

map_chain = LLMChain(llm=llm, prompt=prompt)
reduce_chain = StuffDocumentsChain(llm_chain=map_chain, document_variable_name="context")
combine_docs_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name="context"
)

qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_docs_chain)

# --- Ask a Question ---
st.subheader("💬 Ask a Question")
query = st.text_input("Enter your question:")

if query:
    st.markdown("**⏳ Processing your question...**")
    start = time.time()
    answer = qa_chain.invoke({"query": query})
    st.markdown("**🧠 Answer:**")
    st.write(answer["result"])
    st.caption(f"⏱️ Response generated in {time.time() - start:.2f} seconds")

    st.markdown("**🔍 Retrieved Chunks:**")
    for doc in retriever.get_relevant_documents(query):
        st.code(doc.page_content)

# --- t-SNE Visualization ---
st.subheader("🧬 Embedding Visualization (t-SNE)")
if len(docs) >= 2:
    st.write("📊 Computing embeddings for t-SNE...")
    start = time.time()
    vecs = embeddings.embed_documents([doc.page_content for doc in docs])
    emb_array = np.array(vecs)
    tsne = TSNE(n_components=2, perplexity=min(30, len(docs) - 1), random_state=42)
    tsne_result = tsne.fit_transform(emb_array)
    st.write(f"✅ t-SNE completed in {time.time() - start:.2f}s")

    fig, ax = plt.subplots()
    ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
    ax.set_title("t-SNE of Document Embeddings")
    st.pyplot(fig)
else:
    st.warning("You need at least 2 chunks to show t-SNE.")
