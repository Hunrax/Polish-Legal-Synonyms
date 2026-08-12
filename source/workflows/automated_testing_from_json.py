import csv
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_as_a_judge.llm_pairwise_judge import evaluate_synonyms_with_llm
from metrics.metrics_pairwise import calculate_metrics

ITERATIONS = 30

INPUT_DIR = Path("input_hybrid")


def load_pairs_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []

    for item in data.get("results", []):
        pair = item.get("pair")

        if not pair or len(pair) != 2:
            continue

        pairs.append(tuple(pair))

    return pairs


def pairs_to_groups(pairs):
    return [set(pair) for pair in pairs]


def run_json_testing(json_path):
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    pairs = load_pairs_from_json(json_path)

    print(f"Loaded {len(pairs)} pairs from JSON.")

    groups = pairs_to_groups(pairs)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_dir = Path("automated_tests_results_json30")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_file = (
        results_dir
        / f"json_eval_{json_path.stem}_{timestamp}.csv"
    )

    file_exists = csv_file.exists()

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "json_file",
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

            evaluation = evaluate_synonyms_with_llm(
                groups,
                label=(
                    f"json_eval_"
                    f"{json_path.stem}_"
                    f"{timestamp}_"
                    f"iteration_{iteration}"
                ),
            )

            unique_words = set()

            for pair in pairs:
                unique_words.update(pair)

            lemmas_count = len(unique_words)

            metrics = calculate_metrics(
                evaluation,
                lemmas_count,
                lemmas_count,
            )

            result = {
                "json_file": json_path.name,
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

    if not INPUT_DIR.exists():
        print(f"Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    json_files = sorted(INPUT_DIR.glob("*.json"))

    if not json_files:
        print("No JSON files found in /input")
        sys.exit(1)

    for json_path in json_files:

        print("\n========================================")
        print(f"STARTING: {json_path.name}")
        print("========================================\n")

        run_json_testing(json_path)