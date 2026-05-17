import sys
from pathlib import Path

# Add the workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from text_extraction.extract_from_pdf import clean_and_extract, load_pdf_text, extract_words_without_lemmas
from models.bert.bert import get_contextual_embeddings, get_embeddings
from clustering.dbscan import cluster_synonyms
from gold_standard.load_gold_standard import load_groups_from_file
from metrics.metrics import calculate_metrics
from text_extraction.lemmatization import lemmatize_groups
from llm_as_a_judge.llm_pairwise_judge import evaluate_synonyms_with_llm

def run_workflow(pdf_file):
    raw_text = load_pdf_text(pdf_file)
    all_words = extract_words_without_lemmas(pdf_file)

    print(f"Extracted {len(all_words)} unique words.")

    word_embeddings = get_contextual_embeddings([raw_text], all_words)
    print(f"Generated embeddings for {len(word_embeddings)} words.\n")

    clusters = cluster_synonyms(word_embeddings, eps=0.45, min_samples=2)
    print(f"Identified {len(clusters)} synonym clusters.\n")

    for cluster_id, words in clusters.items():
        print(f"Cluster {cluster_id}: {', '.join(words)}")
    predicted_groups = list(clusters.values())
    predicted_groups = lemmatize_groups(predicted_groups)

    for group in predicted_groups:
        print(f"Predicted group: {', '.join(group)}")

    evaluation = evaluate_synonyms_with_llm(predicted_groups, label="bert_dbscan")
    print(f"LLM Evaluation: {evaluation}")

if __name__ == "__main__":
    run_workflow("text_extraction/input/orzeczenie_7.pdf")
