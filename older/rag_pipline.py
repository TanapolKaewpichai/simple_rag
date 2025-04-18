from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os

# 1. Load text
loader = TextLoader("data.txt")
documents = loader.load()

# 2. Split
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Vector store
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# 5. Use transformers pipeline (offline, no API required)
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256,
    temperature=0.2,
)

llm = HuggingFacePipeline(pipeline=qa_pipeline)

# 6. QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 7. Ask your question
query = "What is this document about?"
result = qa_chain({"query": query})

print("Answer:", result["result"])
