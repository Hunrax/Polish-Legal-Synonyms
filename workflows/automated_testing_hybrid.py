import csv
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from text_extraction.extract_from_pdf import clean_and_extract
from llm_as_a_judge.llm_pairwise_judge import evaluate_synonyms_with_llm
from metrics.metrics_pairwise import calculate_metrics
from models.plwordnet_fasttext_hybrid.plwordnet_fasttext_hybrid import (
    group_similar_words_hybrid,
)

THRESHOLD = 0.8
ITERATIONS = 1

def run_automated_testing(pdf_filename):
    pdf_path = Path("input") / pdf_filename

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return

    lemmas = clean_and_extract(pdf_path).keys()

    print(f"Extracted {len(lemmas)} unique lemmas.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_dir = Path("automated_tests_results_hybrid")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_file = (results_dir/ f"automated_results_hybrid_{Path(pdf_filename).stem}_{timestamp}.csv")

    file_exists = Path(csv_file).exists()

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdf",
                "iteration",
                "average_score",
                "median_score",
                "acceptance_rate",
                "coverage",
            ],
        )

        if not file_exists:
            writer.writeheader()

        for iteration in range(1, ITERATIONS + 1):
            print("\n====================================")
            print(f"ITERATION {iteration}/{ITERATIONS}")
            print("====================================\n")

            groups = group_similar_words_hybrid(lemmas, threshold=THRESHOLD)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            evaluation = evaluate_synonyms_with_llm(groups, label=f"hybrid_file_{Path(pdf_filename).stem}_{timestamp}_iteration_{iteration}")

            lemmas_in_pairs = len(set(word for group in groups for word in group))

            metrics = calculate_metrics(evaluation, len(lemmas), lemmas_in_pairs)

            result = {
                "pdf": pdf_filename,
                "iteration": iteration,
                "average_score": metrics["average_score"],
                "median_score": metrics["median_score"],
                "acceptance_rate": metrics["acceptance_rate"],
                "coverage": metrics["coverage"],
            }

            print("\n--- Metrics summary ---")
            print(f"Average score: {metrics['average_score']:.2f}")
            print(f"Median score: {metrics['median_score']}")
            print(f"Acceptance rate: {metrics['acceptance_rate']:.2%}")
            print(f"Coverage: {metrics['coverage']:.2%}")

            writer.writerow(result)
            f.flush()

    print(f"\nSaved results to {csv_file}")


if __name__ == "__main__":
    for i in range(7, 11):
        pdf_filename = f"orzeczenie_{i}.pdf"
        # pdf_filename = f"ustawa_{i}.pdf"

        print("\n========================================")
        print(f"STARTING: {pdf_filename}")
        print("========================================\n")

        run_automated_testing(pdf_filename)