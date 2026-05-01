import os
import chromadb
import shutil
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
DB_FOLDER = "chroma_db"
COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents(folder_path):
    documents = []
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents.append({
                    "id": filename,
                    "text": content
                })
    return documents

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main():
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Connectong to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_FOLDER)
    
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    documents = load_documents(DATA_FOLDER)
    
    id = []
    texts = []
    metadatas = []
    embeddings = []
    
    chunk_id = 0
    
    for doc in documents:
        chunks = split_text(doc["text"])
        
        for chunk in chunks:
            id.append(f"{doc['id']}_{chunk_id}")
            texts.append(chunk)
            metadatas.append({"source": doc["id"]})
            
            embedding = embedding_model.encode(chunk).tolist()
            embeddings.append(embedding)
            chunk_id += 1
        
        print(f"Adding {len(chunks)} chunks from {doc['id']} to ChromaDB...")
        
    collection.add(
        ids=id,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print("Ingestion complete.")
    
if __name__ == "__main__":
    main()