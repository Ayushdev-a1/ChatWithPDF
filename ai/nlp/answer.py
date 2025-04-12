import requests

def get_answer(question, context):
    """Fetch answer from Ollama based on context."""
    ollama_url = "http://localhost:11434/api/generate"  # Adjust if different
    payload = {
        "model": "llama3",
        "prompt": f"Context: {context}\nQuestion: {question}\nAnswer:",
        "stream": False
    }
    try:
        response = requests.post(ollama_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "No answer")
    except requests.RequestException as e:
        return f"Error: {e}"

if __name__ == "__main__":
    from vector_store import VectorStore
    from embeddings import get_embeddings
    from pdf_processing.extract import extract_and_chunk
    chunks = extract_and_chunk("data/pdfs/sample1.pdf")
    embeddings = get_embeddings(chunks)
    store = VectorStore()
    store.add(embeddings, chunks)
    question = "What’s the main point?"
    query_embedding = get_embeddings([question])[0]
    context = store.search(query_embedding)[0]
    print(get_answer(question, context))