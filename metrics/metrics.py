from itertools import combinations


def get_pairs(groups):
    """Zamienia listę grup na zbiór unikalnych par słów."""
    pairs = set()
    for group in groups:
        for pair in combinations(sorted(group), 2):
            pairs.add(pair)
    return pairs


def calculate_metrics(predicted_groups, gold_groups):
    pred_pairs = get_pairs(predicted_groups)
    gold_pairs = get_pairs(gold_groups)
    
    tp = len(pred_pairs.intersection(gold_pairs))
    
    precision = tp / len(pred_pairs) if len(pred_pairs) > 0 else 0.0
    
    recall = tp / len(gold_pairs) if len(gold_pairs) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp_count": tp,
        "pred_pairs_count": len(pred_pairs),
        "gold_pairs_count": len(gold_pairs)
    }