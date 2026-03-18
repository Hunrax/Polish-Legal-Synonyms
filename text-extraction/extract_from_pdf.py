import fitz
import spacy
import re
import os
from pyMorfologik import Morfologik
from pyMorfologik.parsing import ListParser

def clean_and_extract(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return None
    
    try:
        nlp = spacy.load("pl_core_news_lg")
    except OSError:
        print("Error while loading 'pl_core_news_lg'. Run: python -m spacy download pl_core_news_lg")
        return None
    
    morf = Morfologik()
    parser = ListParser()

    try:
        doc_pdf = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return None

    raw_text = ""
    for page in doc_pdf:
        page_text = page.get_text()

        page_text = page_text.replace("-\n", "")

        page_text = page_text.replace("\n", " ")
        raw_text += " " + page_text
    
    raw_text = re.sub(r"\"", "", raw_text)
    raw_text = re.sub(r"\d+", "", raw_text)
    raw_text = re.sub(r"[\"§\.\,\:\(\)]", " ", raw_text)

    doc = nlp(raw_text)

    allowed_pos = {"NOUN", "VERB", "ADJ", "ADV"}

    lemma_map = {}

    for token in doc:
        word = token.text.lower()

        if token.pos_ in allowed_pos and token.is_alpha and len(word) >= 3:
            suggestions = morf.stem([word], parser)
            
            if suggestions and suggestions[0][1]:
                lemma_dict = suggestions[0][1]
                first_lemma = list(lemma_dict.keys())[0].lower()
                print(f"Original: {word} -> Lemma: {first_lemma}")
                
                if first_lemma not in lemma_map:
                    lemma_map[first_lemma] = set()
                
                lemma_map[first_lemma].add(word)

    return lemma_map


if __name__ == "__main__":
    print("Extract lemmas and words from a PDF file.")
    input_pdf = input("Enter the path to the input PDF file (e.g. ustawa_1.pdf): ").strip()

    mapping = clean_and_extract(input_pdf)

    if mapping:
        sorted_lemmas = sorted(mapping.keys())
        
        all_extracted_words = set()
        for words in mapping.values():
            all_extracted_words.update(words)
        sorted_originals = sorted(list(all_extracted_words))

        with open("lemmas.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sorted_lemmas))
        
        with open("extracted_words.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sorted_originals))

        with open("lemma_mapping.txt", "w", encoding="utf-8") as f:
            for lemma in sorted_lemmas:
                originals = ", ".join(sorted(list(mapping[lemma])))
                f.write(f"{lemma}: [{originals}]\n")

    print("\n--- PROCESS COMPLETED ---")
    print(f"1. Extracted and saved {len(sorted_originals)} original words to file 'extracted_words.txt'.")
    print(f"2. Reduced to {len(sorted_lemmas)} unique lemmas and saved to file 'lemmas.txt'.")
    print(f"3. Full mapping of relations saved to file 'lemma_mapping.txt'.")