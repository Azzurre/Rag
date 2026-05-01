import chromadb
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer


DB_FOLDER = "chroma_db"
COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path=DB_FOLDER)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve_relevant_chunks(question, top_k=3):
    embedding_model = load_embedding_model()
    collection = load_chroma_collection()

    question_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return chunks, metadatas, distances


def build_prompt(question, chunks, metadatas):
    context_parts = []

    for i, chunk in enumerate(chunks):
        source = metadatas[i].get("source", "unknown")
        context_parts.append(f"Source: {source}\nContent: {chunk}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are FightIQ, a combat sports knowledge assistant.

You answer questions about MMA, boxing, kickboxing, Muay Thai, grappling, fight training, conditioning, and combat sports strategy.

Use ONLY the provided context below.
Do not invent techniques, medical advice, or unsafe training instructions.
If the answer is not in the context, say:
"I do not have enough information in the provided fight knowledge base."

Give practical, clear explanations.
When useful, structure the answer as:
1. What it means
2. Why it matters
3. How it is applied in training or fighting

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def ask_llm(prompt):
    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def create_preview(text, max_length=220):
    cleaned = " ".join(text.split())

    if len(cleaned) <= max_length:
        return cleaned

    return cleaned[:max_length] + "..."


def answer_question(question):
    chunks, metadatas, distances = retrieve_relevant_chunks(question)
    prompt = build_prompt(question, chunks, metadatas)
    answer = ask_llm(prompt)

    sources = []

    for index, metadata in enumerate(metadatas):
        source = metadata.get("source", "unknown")
        preview = create_preview(chunks[index])
        distance = distances[index]

        sources.append({
            "source": source,
            "preview": preview,
            "distance": distance
        })

    return answer, sources


def main():
    st.set_page_config(
        page_title="FightIQ",
        page_icon="🥊",
        layout="wide"
    )

    st.title("🥊 FightIQ")
    st.caption("A local RAG assistant for MMA, boxing, kickboxing, Muay Thai, grappling, and fight training.")

    with st.sidebar:
        st.header("About")
        st.write(
            "FightIQ answers questions using your local combat sports knowledge base. "
            "It retrieves relevant chunks from ChromaDB and uses a local Ollama model to generate grounded answers."
        )

        st.divider()

        st.subheader("Example questions")
        st.write("- What is ring cutting in boxing?")
        st.write("- How do I defend in the Muay Thai clinch?")
        st.write("- Why is the jab important?")
        st.write("- How should a beginner structure MMA training?")

        st.divider()

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Sources used"):
                    for i, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**{i}. {source['source']}**")
                        st.write(f"Relevance distance: `{source['distance']:.4f}`")
                        st.write(source["preview"])

    user_question = st.chat_input("Ask FightIQ something...")

    if user_question:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving fight knowledge and generating answer..."):
                answer, sources = answer_question(user_question)

            st.write(answer)

            with st.expander("Sources used"):
                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**{i}. {source['source']}**")
                    st.write(f"Relevance distance: `{source['distance']:.4f}`")
                    st.write(source["preview"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


if __name__ == "__main__":
    main()