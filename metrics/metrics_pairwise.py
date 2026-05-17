import statistics    

def calculate_metrics(data: dict, all_lemmas: int, lemmas_in_pairs: int):
    valid_scores = [r["value"] for r in data["results"] if r["value"] is not None]

    average = (
        round(sum(valid_scores) / len(valid_scores), 2)
        if valid_scores
        else None
    )

    median = (
        round(statistics.median(valid_scores), 2)
        if valid_scores
        else None
    )
    score_distribution = {str(i): valid_scores.count(i) for i in range(1, 6)}

    accepted_scores = [s for s in valid_scores if s >= 4]
    acceptance_rate = round(len(accepted_scores) / len(valid_scores), 2) if valid_scores else None

    coverage = round(lemmas_in_pairs / all_lemmas, 2) if all_lemmas > 0 else None

    return {
        "average_score": average,
        "median_score": median,
        "score_distribution": score_distribution,
        "acceptance_rate": acceptance_rate,
        "coverage": coverage,
    }

