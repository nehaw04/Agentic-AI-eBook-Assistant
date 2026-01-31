import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Load environment variables
load_dotenv()

# We create a small helper class to force the dimensions to 512
class SlicedGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts):
        vectors = super().embed_documents(texts)
        return [v[:512] for v in vectors] # Cuts each vector to 512

    def embed_query(self, text):
        vector = super().embed_query(text)
        return vector[:512] # Cuts the search query to 512

def ingest_docs():
    # 2. Load and Split
    pdf_path = "./data/ebook.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Prepared {len(chunks)} chunks.")

    # 3. Initialize our 'Fixed' Embeddings
    embeddings = SlicedGeminiEmbeddings(
        model="models/text-embedding-004",
        task_type="retrieval_document"
    )

    # 4. Upload to Pinecone
    index_name = os.getenv("PINECONE_INDEX_NAME")
    print(f"Uploading to {index_name}...")
    
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    print("Success! Ingestion complete.")

if __name__ == "__main__":
    ingest_docs()