from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load new document
loader = TextLoader("new_info.txt")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# 3. Embed using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Connect to Milvus
vector_db = Milvus(
    embedding_function=embeddings,
    collection_name="langchain_docs",  # same collection name used before
    connection_args={"host": "localhost", "port": "19530"}
)

# 5. Add new vectors
vector_db.add_documents(chunks)

print("✅ New documents indexed into vector DB.")
