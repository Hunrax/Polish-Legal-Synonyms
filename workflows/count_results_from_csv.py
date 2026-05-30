import csv
import json
from pathlib import Path
from statistics import mean

INPUT_DIR = Path("automated_tests_results_json30")
OUTPUT_DIR = Path("aggregated_results_json30")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_csv(csv_path):
    average_scores = []
    median_scores = []
    acceptance_rates = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            average_scores.append(float(row["average_score"]))
            median_scores.append(float(row["median_score"]))
            acceptance_rates.append(float(row["acceptance_rate"]))

    if not average_scores:
        print(f"No data in {csv_path.name}")
        return None

    return {
        "file": csv_path.name,
        "iterations": len(average_scores),

        "average_score": {
            "mean": round(mean(average_scores), 4),
        },

        "median_score": {
            "mean": round(mean(median_scores), 4),
        },

        "acceptance_rate": {
            "mean": round(mean(acceptance_rates), 4),
        },
    }


def save_json(result, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    if not INPUT_DIR.exists():
        print(f"Directory does not exist: {INPUT_DIR}")
        exit(1)

    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        exit(1)

    all_results = []

    for csv_path in csv_files:

        print("\n====================================")
        print(f"PROCESSING: {csv_path.name}")
        print("====================================")

        result = aggregate_csv(csv_path)

        if result is None:
            continue

        all_results.append(result)

        output_path = OUTPUT_DIR / f"{csv_path.stem}_summary.json"
        save_json(result, output_path)

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nSaved summary to: {output_path}")

    # ==========================
    # GLOBAL: only mean of means
    # ==========================

    if all_results:

        global_summary = {
            "files_processed": len(all_results),

            "average_score_mean": round(
                mean(r["average_score"]["mean"] for r in all_results),
                4
            ),

            "median_score_mean": round(
                mean(r["median_score"]["mean"] for r in all_results),
                4
            ),

            "acceptance_rate_mean": round(
                mean(r["acceptance_rate"]["mean"] for r in all_results),
                4
            ),
        }

        global_output_path = OUTPUT_DIR / "global_summary.json"

        save_json(global_summary, global_output_path)

        print("\n====================================")
        print("GLOBAL SUMMARY")
        print("====================================")

        print(json.dumps(global_summary, indent=2, ensure_ascii=False))
        print(f"\nSaved global summary to: {global_output_path}")