import sys
from pathlib import Path

# Add the workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from text_extraction.extract_from_pdf import clean_and_extract, load_pdf_text
from models.bert.bert import get_contextual_embeddings, get_embeddings
from clustering.dbscan import cluster_synonyms

def run_workflow(pdf_file):
    raw_text = load_pdf_text(pdf_file)
    lemma_map = clean_and_extract(pdf_file)

    sorted_lemmas = sorted(lemma_map.keys())
    all_words = set()
    for words in lemma_map.values():
        all_words.update(words)

    print(f"Extracted {len(sorted_lemmas)} unique lemmas and {len(all_words)} unique words.")

    word_embeddings = get_contextual_embeddings([raw_text], set(all_words))
    print(f"Generated embeddings for {len(word_embeddings)} words.")

    clusters = cluster_synonyms(word_embeddings, eps=0.4, min_samples=2)
    print(f"Identified {len(clusters)} synonym clusters.")

    for cluster_id, words in clusters.items():
        print(f"Cluster {cluster_id}: {', '.join(words)}")

if __name__ == "__main__":
    run_workflow("text_extraction/input/orzeczenie_1.pdf")
