from ai.nlp.embeddings import get_embeddings
from ai.nlp.vector_store import VectorStore

def test_embeddings():
    embeddings = get_embeddings(["test text"])
    assert embeddings.shape[0] == 1, "Embedding generation failed"
    print("Embeddings test passed")

def test_vector_store():
    store = VectorStore()
    embeddings = get_embeddings(["test1", "test2"])
    store.add(embeddings, ["test1", "test2"])
    result = store.search(get_embeddings(["test1"])[0])[0]
    assert result == "test1", "Vector search failed"
    print("Vector store test passed")

if __name__ == "__main__":
    test_embeddings()
    test_vector_store()