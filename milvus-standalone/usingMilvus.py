import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA

from pymilvus import connections, utility

# --- Environment Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_NO_PYTHON_EXTENSION_WARNING"] = "1"

st.set_page_config(page_title="Local RAG with Milvus", layout="wide")
st.title("🧠 Local RAG Q&A - MapReduce + Milvus")

# --- Load Documents ---
@st.cache_resource(show_spinner=False)
def load_docs():
    loader = TextLoader("data.txt")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    return splitter.split_documents(raw_docs)

docs = load_docs()
st.success(f"✅ Loaded {len(docs)} chunks from data.txt")

# --- Connect to Milvus and Prepare Vector DB ---
@st.cache_resource(show_spinner=False)
def get_vector_db(_docs):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        connections.connect(alias="default", host="milvus-standalone", port="19530")
        if utility.has_collection("langchain_docs"):
            st.info("ℹ️ Found existing Milvus collection: langchain_docs")
            vector_db = Milvus(
                embedding_function=embeddings,
                collection_name="langchain_docs",
                connection_args={"host": "milvus-standalone", "port": "19530"},
            )
        else:
            st.info("⚙️ Creating new Milvus collection: langchain_docs")
            vector_db = Milvus.from_documents(
                _docs,
                embedding=embeddings,
                collection_name="langchain_docs",
                connection_args={"host": "milvus-standalone", "port": "19530"},
                drop_old=True
            )
            utility.flush(["langchain_docs"])
            st.success("✅ Vector data saved to Milvus")

        return vector_db, embeddings

    except Exception as e:
        st.error(f"❌ Failed to connect or upload to Milvus: {e}")
        st.stop()

vector_db, embeddings = get_vector_db(docs)

# --- Retriever Setup ---
dense_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3
retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.7, 0.3])

# --- LLM Setup ---
qa_pipe = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=qa_pipe)

prompt = PromptTemplate.from_template(
    "You are a helpful assistant.\n"
    "Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Helpful Answer:"
)

map_chain = LLMChain(llm=llm, prompt=prompt)
reduce_chain = StuffDocumentsChain(llm_chain=map_chain, document_variable_name="context")

combine_docs_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name="context"
)

qa_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=combine_docs_chain
)

# --- Ask a Question ---
st.subheader("💬 Ask a Question")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating answer..."):
        answer = qa_chain.invoke({"query": query})
        st.markdown("**🧠 Answer:**")
        st.write(answer["result"])

        st.markdown("**🔍 Retrieved Chunks:**")
        for doc in retriever.get_relevant_documents(query):
            st.code(doc.page_content)

# --- t-SNE Visualization ---
st.subheader("🧬 Embedding Visualization (t-SNE)")

@st.cache_data(show_spinner=False)
def get_tsne_result(_embeddings, _docs):
    vecs = _embeddings.embed_documents([doc.page_content for doc in _docs])
    emb_array = np.array(vecs)
    tsne = TSNE(n_components=2, perplexity=min(30, len(_docs) - 1), random_state=42)
    return tsne.fit_transform(emb_array)

if len(docs) >= 2:
    tsne_result = get_tsne_result(embeddings, docs)
    fig, ax = plt.subplots()
    ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
    ax.set_title("t-SNE of Document Embeddings")
    st.pyplot(fig)
else:
    st.warning("You need at least 2 chunks to show t-SNE.")
