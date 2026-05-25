import sys
from pathlib import Path

# Add the workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from text_extraction.extract_from_pdf import load_pdf_text, extract_words_without_lemmas
from models.bert.bert import get_contextual_embeddings, load_model_and_tokenizer, model_names
from clustering.dbscan import cluster_synonyms
from text_extraction.lemmatization import lemmatize_groups
from llm_as_a_judge.llm_pairwise_judge import evaluate_synonyms_with_llm
from metrics.metrics_pairwise import calculate_metrics

def run_workflow(pdf_filename, model_name, epsilon, model=None, tokenizer=None):
    pdf_path = Path("input") / pdf_filename

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return

    raw_text = load_pdf_text(pdf_path)
    all_words = extract_words_without_lemmas(pdf_path)

    print(f"Extracted {len(all_words)} unique words.")

    if model is None or tokenizer is None:
        tokenizer, model = load_model_and_tokenizer(model_name)

    word_embeddings = get_contextual_embeddings([raw_text], all_words, model, tokenizer)
    print(f"Generated embeddings for {len(word_embeddings)} words.\n")

    clusters = cluster_synonyms(word_embeddings, eps=epsilon, min_samples=2)
    print(f"Identified {len(clusters)} synonym clusters.\n")

    for cluster_id, words in clusters.items():
        print(f"Cluster {cluster_id}: {', '.join(words)}")
    predicted_groups = list(clusters.values())
    predicted_groups = lemmatize_groups(predicted_groups)

    for group in predicted_groups:
        print(f"Predicted group: {', '.join(group)}")

    evaluation = evaluate_synonyms_with_llm(predicted_groups, label=f"bert_dbscan")

    lemmas_in_pairs = len(set(word for group in predicted_groups for word in group))
    metrics = calculate_metrics(evaluation, len(all_words), lemmas_in_pairs)

    print("\n---Metrics summary---")
    print(f"Average score: {metrics['average_score']:.2f}")
    print(f"Median score: {metrics['median_score']}")
    print(f"Acceptance rate: {metrics['acceptance_rate']:.2%}")
    print(f"Coverage: {metrics['coverage']:.2%}")
    print("Score distribution:")
    for score, count in metrics["score_distribution"].items():
        print(f"  {score}: {count}")

    return metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_bert_dbscan.py <pdf_filename> [model_name] [epsilon]")
        sys.exit(1)

    pdf_filename = sys.argv[1]

    # If model_name is provided as argument, use it
    if len(sys.argv) >= 3:
        model_name = sys.argv[2]
        if model_name not in model_names:
            print(f"Error: Model '{model_name}' not found.")
            print("Available models:")
            for name in model_names:
                print(f"  - {name}")
            sys.exit(1)
    else:
        print("Available models:")
        for i, name in enumerate(model_names, start=1):
            print(f"{i}. {name}")
        model_choice = int(input("Enter the number of your choice: "))
        model_name = model_names[model_choice - 1]
        print(f"Selected model: {model_name}\n")

    # If epsilon is provided as argument, use it
    if len(sys.argv) >= 4:
        epsilon = float(sys.argv[3])
    else:
        epsilon = float(input("Enter DBSCAN epsilon (e.g., 0.4): "))

    run_workflow(pdf_filename, model_name, epsilon)
