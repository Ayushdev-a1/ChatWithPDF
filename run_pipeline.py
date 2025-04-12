from ai.pdf_processing.extract import extract_and_chunk
from ai.nlp.embeddings import get_embeddings
from ai.nlp.vector_store import VectorStore
from ai.nlp.answer import get_answer

def run_pipeline(pdf_path, question):
    chunks = extract_and_chunk(pdf_path)
    embeddings = get_embeddings(chunks)
    store = VectorStore()
    store.add(embeddings, chunks)
    query_embedding = get_embeddings([question])[0]
    context = store.search(query_embedding)[0]
    return get_answer(question, context)

if __name__ == "__main__":
    question = "What’s the main point?"
    result = run_pipeline("data/pdfs/sample1.pdf", question)
    print(f"Answer: {result}")