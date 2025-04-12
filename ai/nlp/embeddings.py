from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(text_chunks):
    """Convert text chunks to embeddings."""
    return model.encode(text_chunks, convert_to_tensor=False)

if __name__ == "__main__":
    from pdf_processing.extract import extract_and_chunk
    chunks = extract_and_chunk("data/pdfs/sample1.pdf")
    embeddings = get_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")