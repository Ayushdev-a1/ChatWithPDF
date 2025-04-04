
from ai.pdf_processing.extract import extract_text
from ai.nlp.tokenize import get_keywords
from ai.nlp.answer import find_answer

text = extract_text("data/pdfs/sample1.pdf")
keywords = get_keywords("What’s the main point?")
print(find_answer(text, keywords))
