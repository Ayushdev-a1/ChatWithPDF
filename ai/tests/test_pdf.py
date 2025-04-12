from ai.pdf_processing.extract import extract_and_chunk

def test_extraction():
    chunks = extract_and_chunk("data/pdfs/sample1.pdf")
    assert len(chunks) > 0, "No chunks extracted"
    print("PDF extraction test passed")

if __name__ == "__main__":
    test_extraction()