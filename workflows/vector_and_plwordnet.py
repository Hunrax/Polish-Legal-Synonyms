import sys
from pathlib import Path

# Add the workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from text_extraction.extract_from_pdf import clean_and_extract
from models.word2vec.word2vec import group_similar_words_word2vec
from models.fasttext.fasttext import group_similar_words_fasttext
from models.plwordnet.plwordnet import group_similar_words_plwordnet
from models.plwordnet_fasttext_hybrid.plwordnet_fasttext_hybrid import group_similar_words_hybrid
from llm_as_a_judge.llm_pairwise_judge import evaluate_synonyms_with_llm
from metrics.metrics_pairwise import calculate_metrics

model_names = ["word2vec", "fasttext", "plwordnet", "plwordnet_fasttext_hybrid"]

def run_workflow(pdf_filename, model_name, threshold):
    pdf_path = Path("input") / pdf_filename

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return

    lemmas = clean_and_extract(pdf_path).keys()
    print(f"Extracted {len(lemmas)} unique lemmas.")

    if model_name == "word2vec":
        groups = group_similar_words_word2vec(lemmas, threshold=threshold)
        for i, group in enumerate(groups, start=1):
            print(f"Group {i}: {', '.join(group)}")

    elif model_name == "fasttext":
        groups = group_similar_words_fasttext(lemmas, threshold=threshold)
        for i, group in enumerate(groups, start=1):
            print(f"Group {i}: {', '.join(group)}")

    elif model_name == "plwordnet":
        groups = group_similar_words_plwordnet(set(lemmas))
        for i, group in enumerate(groups, start=1):
            print(f"Group {i}: {', '.join(group)}")
    
    elif model_name == "plwordnet_fasttext_hybrid":
        groups = group_similar_words_hybrid(lemmas, threshold=threshold)
        for i, group in enumerate(groups, start=1):
            print(f"Group {i}: {', '.join(group)}")

    evaluation = evaluate_synonyms_with_llm(groups, label=model_name)

    lemmas_in_pairs = len(set(word for group in groups for word in group))
    metrics = calculate_metrics(evaluation, len(lemmas), lemmas_in_pairs)

    print("\n---Metrics summary---")
    print(f"Average score: {metrics['average_score']:.2f}")
    print(f"Median score: {metrics['median_score']}")
    print(f"Acceptance rate: {metrics['acceptance_rate']:.2%}")
    print(f"Coverage: {metrics['coverage']:.2%}")
    print("Score distribution:")
    for score, count in metrics["score_distribution"].items():
        print(f"  {score}: {count}")

    return {
        "model": model_name,
        "threshold": threshold,
        "average_score": metrics["average_score"],
        "median_score": metrics["median_score"],
        "acceptance_rate": metrics["acceptance_rate"],
        "coverage": metrics["coverage"],
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python vector_model.py <pdf_filename>")
        sys.exit(1)

    pdf_filename = sys.argv[1]

    print("Available models:")
    for i, name in enumerate(model_names, start=1):
        print(f"{i}. {name}")
    model_choice = int(input("Enter the number of your choice: "))
    model_name = model_names[model_choice - 1]
    print(f"Selected model: {model_name}\n")

    threshold = None
    if(model_name in ["word2vec", "fasttext", "plwordnet_fasttext_hybrid"]):
        threshold = float(input("Enter similarity threshold (e.g., 0.5): "))

    run_workflow(pdf_filename, model_name, threshold)