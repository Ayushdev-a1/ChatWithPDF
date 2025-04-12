import pdfplumber

def chunk_text(text, chunk_size=500):
    """Split text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def extract_and_chunk(pdf_path):
    """Extract text from PDF and chunk it."""
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return chunk_text(full_text)

if __name__ == "__main__":
    chunks = extract_and_chunk("data/pdfs/sample1.pdf")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")