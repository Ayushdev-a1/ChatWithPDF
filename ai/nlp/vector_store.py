import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def add(self, embeddings, texts):
        """Add embeddings and corresponding texts to the store."""
        self.chunks = texts
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

    def search(self, query_embedding, k=1):
        """Search for the k nearest neighbors."""
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.chunks[i] for i in indices[0]]

if __name__ == "__main__":
    from embeddings import get_embeddings
    from pdf_processing.extract import extract_and_chunk
    chunks = extract_and_chunk("data/pdfs/sample1.pdf")
    embeddings = get_embeddings(chunks)
    store = VectorStore()
    store.add(embeddings, chunks)
    print("Vector store initialized")