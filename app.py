import uuid
from datetime import datetime

import chromadb
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

from web_search_ingest import search_and_extract_web_documents


DB_FOLDER = "chroma_db"
COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2"

# Lower = stricter, higher = more likely to trust local database.
# If FightIQ searches the web too much, increase this to 1.1 or 1.2.
# If FightIQ does not search web enough, decrease this to 0.7 or 0.8.
LOCAL_CONFIDENCE_DISTANCE_THRESHOLD = 0.9


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path=DB_FOLDER)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def retrieve_relevant_chunks(question, top_k=3):
    embedding_model = load_embedding_model()
    collection = load_chroma_collection()

    question_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    return chunks, metadatas, distances


def build_prompt(question, chunks, metadatas):
    context_parts = []

    for i, chunk in enumerate(chunks):
        source = metadatas[i].get("source", "unknown")
        url = metadatas[i].get("url")

        source_text = f"Source: {source}"
        if url:
            source_text += f"\nURL: {url}"

        context_parts.append(f"{source_text}\nContent: {chunk}")

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

def should_force_web_search(question):
    live_keywords = [
        "next ufc",
        "upcoming ufc",
        "ufc card",
        "fight card",
        "who is fighting",
        "who's fighting",
        "latest",
        "today",
        "tonight",
        "tomorrow",
        "this weekend",
        "schedule",
        "results",
        "rankings",
        "who won",
        "last night"
    ]

    question_lower = question.lower()

    return any(keyword in question_lower for keyword in live_keywords)


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


def add_web_documents_to_chroma(documents, max_chunks_per_doc=5):
    """
    Takes documents returned by web_search_ingest.py and adds them directly to ChromaDB.
    Expected document format:
    {
        "title": "...",
        "url": "...",
        "text": "..."
    }
    """

    if not documents:
        return {
            "chunks_added": 0,
            "sources": []
        }

    embedding_model = load_embedding_model()
    collection = load_chroma_collection()

    total_chunks_added = 0
    learned_sources = []

    for document in documents:
        title = document.get("title", "Unknown web source")
        url = document.get("url", "")
        text = document.get("text", "")

        if not text.strip():
            continue

        chunks = split_text(text)

        ids = []
        documents_to_add = []
        metadatas = []
        embeddings = []

        for chunk in chunks[:max_chunks_per_doc]:
            chunk_id = f"web_{uuid.uuid4()}"

            ids.append(chunk_id)
            documents_to_add.append(chunk)

            metadatas.append({
                "source": title,
                "url": url,
                "source_type": "web",
                "learned_at": datetime.utcnow().isoformat()
            })

            embedding = embedding_model.encode(chunk).tolist()
            embeddings.append(embedding)

        if ids:
            collection.add(
                ids=ids,
                documents=documents_to_add,
                metadatas=metadatas,
                embeddings=embeddings
            )

            total_chunks_added += len(ids)

            learned_sources.append({
                "title": title,
                "url": url,
                "chunks_added": len(ids)
            })

    return {
        "chunks_added": total_chunks_added,
        "sources": learned_sources
    }


def answer_question(question):
    chunks, metadatas, distances = retrieve_relevant_chunks(question)

    best_distance = distances[0] if distances else 999
    learned_from_web = None

    if should_force_web_search(question) or best_distance > LOCAL_CONFIDENCE_DISTANCE_THRESHOLD:
        web_documents = search_and_extract_web_documents(
            query=question,
            max_results=3
        )

        learned_from_web = add_web_documents_to_chroma(web_documents)

        if learned_from_web["chunks_added"] > 0:
            chunks, metadatas, distances = retrieve_relevant_chunks(question)

    if not chunks:
        return (
            "I do not have enough information in the provided fight knowledge base.",
            [],
            learned_from_web
        )

    prompt = build_prompt(question, chunks, metadatas)
    answer = ask_llm(prompt)

    sources = []

    for index, metadata in enumerate(metadatas):
        source = metadata.get("source", "unknown")
        url = metadata.get("url", None)
        source_type = metadata.get("source_type", "local")
        preview = create_preview(chunks[index])
        distance = distances[index] if index < len(distances) else 999

        sources.append({
            "source": source,
            "url": url,
            "source_type": source_type,
            "preview": preview,
            "distance": distance
        })

    return answer, sources, learned_from_web


def display_sources(sources):
    if not sources:
        st.write("No sources found.")
        return

    for i, source in enumerate(sources, start=1):
        st.markdown(f"**{i}. {source['source']}**")
        st.write(f"Type: `{source['source_type']}`")
        st.write(f"Relevance distance: `{source['distance']:.4f}`")

        if source.get("url"):
            st.write(source["url"])

        st.write(source["preview"])


def main():
    st.set_page_config(
        page_title="FightIQ",
        page_icon="🥊",
        layout="wide"
    )

    st.title("🥊 FightIQ")
    st.caption(
        "A local-first RAG assistant for MMA, boxing, kickboxing, Muay Thai, "
        "grappling, and fight training."
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "FightIQ answers questions using your local combat sports knowledge base. "
            "If the local database does not have a strong match, it can search the web, "
            "learn from extracted web sources, and answer using the updated database."
        )

        st.divider()

        st.subheader("Example questions")
        st.write("- What is ring cutting in boxing?")
        st.write("- How do I defend in the Muay Thai clinch?")
        st.write("- Why is the jab important?")
        st.write("- How should a beginner structure MMA training?")
        st.write("- How do I defend a calf kick?")

        st.divider()

        st.subheader("Settings")
        st.write(f"Local confidence threshold: `{LOCAL_CONFIDENCE_DISTANCE_THRESHOLD}`")
        st.caption("Lower distance means a stronger match.")

        st.divider()

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":
                if message.get("learned_from_web") and message["learned_from_web"]["chunks_added"] > 0:
                    st.info(
                        f"FightIQ searched the web and learned from "
                        f"{len(message['learned_from_web']['sources'])} source(s)."
                    )

                if "sources" in message:
                    with st.expander("Sources used"):
                        display_sources(message["sources"])

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
                answer, sources, learned_from_web = answer_question(user_question)

            st.write(answer)

            if learned_from_web and learned_from_web["chunks_added"] > 0:
                st.info(
                    f"FightIQ searched the web and learned from "
                    f"{len(learned_from_web['sources'])} source(s)."
                )

            with st.expander("Sources used"):
                display_sources(sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "learned_from_web": learned_from_web
        })


if __name__ == "__main__":
    main()