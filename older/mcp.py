import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Optional retriever enhancement
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- Environment Fix ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_NO_PYTHON_EXTENSION_WARNING"] = "1"

# --- Streamlit UI ---
st.title("🧠 Local RAG Q&A - MapReduce + FAISS")

# Load and chunk your documents
loader = TextLoader("data.txt")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = splitter.split_documents(raw_docs)

# Embedding & Vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
dense_retriever = db.as_retriever(search_kwargs={"k": 3})

# Optional BM25 + Hybrid Retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3
retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.7, 0.3])

# Prompt template
prompt = PromptTemplate.from_template(
    "You are a helpful assistant.\n"
    "Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Helpful Answer:"
)

# LLM setup
qa_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
llm = HuggingFacePipeline(pipeline=qa_pipe)

# MapReduce Chain
map_chain = LLMChain(llm=llm, prompt=prompt)
reduce_chain = StuffDocumentsChain(llm_chain=map_chain, document_variable_name="context")
combine_docs_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name="context"
)

# Final QA chain
qa_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=combine_docs_chain
)

# --- Ask your question ---
st.subheader("💬 Ask a question")
query = st.text_input("Ask your question:")

if query:
    answer = qa_chain.invoke({"query": query})
    st.markdown("**🧠 Answer:**")
    st.write(answer["result"])

    st.markdown("**🔍 Retrieved Chunks:**")
    for doc in retriever.get_relevant_documents(query):
        st.code(doc.page_content)

# --- Optional: Visualize Embeddings ---
st.subheader("🧬 Embedding Visualization (t-SNE)")
if len(docs) >= 2:
    vecs = embeddings.embed_documents([doc.page_content for doc in docs])
    emb_array = np.array(vecs)
    tsne = TSNE(n_components=2, perplexity=min(30, len(docs)-1), random_state=42)
    tsne_result = tsne.fit_transform(emb_array)

    fig, ax = plt.subplots()
    ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
    ax.set_title("t-SNE of Document Embeddings")
    st.pyplot(fig)
else:
    st.warning("You need at least 2 chunks to show t-SNE.")
