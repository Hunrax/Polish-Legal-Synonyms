import csv
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from text_extraction.extract_from_pdf import clean_and_extract
from models.word2vec.word2vec import group_similar_words_word2vec
from models.fasttext.fasttext import group_similar_words_fasttext
from models.plwordnet.plwordnet import group_similar_words_plwordnet
from llm_as_a_judge.llm_pairwise_judge import evaluate_synonyms_with_llm
from metrics.metrics_pairwise import calculate_metrics


def run_automated_testing(pdf_filename):
    pdf_path = Path("input") / pdf_filename

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return

    models = ["plwordnet", "fasttext", "word2vec"]

    start = 0.2
    end = 0.8
    step = 0.02

    lemmas = clean_and_extract(pdf_path).keys()
    print(f"Extracted {len(lemmas)} unique lemmas.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    csv_file = f"automated_results_{Path(pdf_filename).stem}_{timestamp}.csv"

    file_exists = Path(csv_file).exists()

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "threshold",
                "average_score",
                "median_score",
                "acceptance_rate",
                "coverage",
            ],
        )

        if not file_exists:
            writer.writeheader()

        for model_name in models:
            print(f"\n==============================")
            print(f"Running model: {model_name}")
            print(f"==============================\n")

            threshold = start

            while threshold <= end:
                print("\n----------------------------")
                print(f"Running threshold = {threshold:.2f}")
                print("----------------------------\n")

                if model_name == "word2vec":
                    groups = group_similar_words_word2vec(lemmas, threshold=threshold)

                elif model_name == "fasttext":
                    groups = group_similar_words_fasttext(lemmas, threshold=threshold)

                elif model_name == "plwordnet":
                    groups = group_similar_words_plwordnet(lemmas)

                evaluation = evaluate_synonyms_with_llm(groups, label=model_name)

                lemmas_in_pairs = len(set(word for group in groups for word in group))
                metrics = calculate_metrics(evaluation, len(lemmas), lemmas_in_pairs)

                result = {
                    "model": model_name,
                    "threshold": threshold,
                    "average_score": metrics["average_score"],
                    "median_score": metrics["median_score"],
                    "acceptance_rate": metrics["acceptance_rate"],
                    "coverage": metrics["coverage"],
                }

                print("\n---Metrics summary---")
                print(f"Average score: {metrics['average_score']:.2f}")
                print(f"Median score: {metrics['median_score']}")
                print(f"Acceptance rate: {metrics['acceptance_rate']:.2%}")
                print(f"Coverage: {metrics['coverage']:.2%}")

                writer.writerow(result)
                f.flush()

                threshold += step

    print(f"\nSaved results to {csv_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python automated_testing_vector_plwordnet.py <pdf_filename>")
        sys.exit(1)

    run_automated_testing(sys.argv[1])