import chromadb
import ollama
from sentence_transformers import SentenceTransformer

DB_FOLDER = "chroma_db"
COLLECTION_NAME = "my_documents"
MODEL_NAME = "llama3.2"

def retrieve_relevant_chunks(question, top_k=3):
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    question_embedding = embedding_model.encode(question).tolist()
    
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["metadatas", "documents"]
    )
    
    chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    return chunks, metadatas

def build_prompt(question, chunks, metadatas):
    context = ""
    
    for i, chunk in enumerate(chunks):
        source = metadatas[i].get('source', 'Unknown')
        context += f"Source: {source}\nContent: {chunk}\n\n"
    
    prompt = f""""
    You are a helpful AI assistant.

    Answer the user's question using ONLY the context below.
    If the answer is not in the context, say:
    "I do not have enough information in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    return prompt

def ask_llm(prompt):
    response = ollama.chat(
        model = MODEL_NAME,
        messages = [
            {
                "role" : "user",
                "content" : prompt 
            }
        ]
    )
    
    print(response)
    
    return response["message"]["content"]

def main():
    print("Welcome to the RAG system!")
    print("Ask a question (or type 'exit' to quit):")
    
    while True:
        question = input("> ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        
        chunks, metadatas = retrieve_relevant_chunks(question)
        prompt = build_prompt(question, chunks, metadatas)
        answer = ask_llm(prompt)
        print(f"Answer: {answer}\n")
        
        print("Sources used:")
        for metadata in metadatas:
            print(f" - {metadata.get('source', 'Unknown')}")
            
        print ("\n" + "-"*50 + "\n")
        
if __name__ == "__main__":
    main()