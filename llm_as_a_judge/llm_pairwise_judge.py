import json
import itertools
import requests
import os
import sys
from pathlib import Path
from datetime import datetime

API_URL = "http://llm-router-service.llm/v1/responses"
MODEL_NAME = "speakleash/Bielik-11B-v2.2-Instruct"

def evaluate_synonym_pair(word1: str, word2: str) -> int:

    SYSTEM_PROMPT = f"""
Oceń, czy podane dwa słowa są synonimami lub wyrazami bliskoznacznymi
w języku polskim, szczególnie w kontekście języka formalnego i prawnego.

Skala ocen:
1 - zdecydowanie nie są synonimami
2 - słabo powiązane znaczeniowo
3 - częściowo podobne znaczeniowo
4 - podobne znaczeniowo, ale nie zawsze wymienne
5 - synonimy lub wyrazy bliskoznaczne

WAŻNE:
- Zwróć TYLKO jedną liczbę od 1 do 5.
- Nie dodawaj żadnego komentarza, uzasadnienia ani tekstu.

Słowo 1: {word1}
Słowo 2: {word2}
"""

    last_error = None

    for attempt in range(1, 11):
        try:
            response = requests.post(
                url=API_URL,
                json={
                    "model": MODEL_NAME,
                    "input": SYSTEM_PROMPT,
                    "max_output_tokens": 1,
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
            )

            response.raise_for_status()
            data = response.json()

            result_text = data["output"][0]["content"][0]["text"].strip()

            if not result_text:
                raise ValueError("Empty response")

            first_char = result_text[0]

            if first_char.isdigit():
                score = int(first_char)
            else:
                raise ValueError(f"Non-numeric response: {result_text}")

            if 1 <= score <= 5:
                return score

            raise ValueError(f"Out of range: {score}")

        except Exception as e:
            last_error = e
            print(f"[retry {attempt}/10] Error for ({word1}, {word2}): {e}")

    raise RuntimeError(f"Failed after 10 attempts for pair ({word1}, {word2}). Last error: {last_error}")

def evaluate_synonyms_with_llm(groups: list, label: str):
    results = []

    if not groups:
        print("No groups to evaluate")
        return {
            "metadata": {
                "model": MODEL_NAME,
                "total_pairs": 0,
                "evaluated_pairs": 0,
            },
            "results": []
        }

    total_pairs = sum(len(list(itertools.combinations(g, 2))) for g in groups)
    done = 0

    for g_idx, group in enumerate(groups):
        pairs = list(itertools.combinations(group, 2))

        for p_idx, (word1, word2) in enumerate(pairs):
            try:
                score = evaluate_synonym_pair(word1, word2)
            except Exception as e:
                print(f"[ERROR] {word1}-{word2}: {e}")
                score = None

            results.append({
                "pair": [word1, word2],
                "value": score,
            })
            done += 1

            print(f"[{done}/{total_pairs}]: {word1} - {word2} => {score}")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        os.makedirs("results", exist_ok=True)
        output_path = Path("results") / f"results_{label}_{timestamp}.json"

    valid_scores = [r["value"] for r in results if r["value"] is not None]

    output_data = {
        "metadata": {
            "model": MODEL_NAME,
            "total_pairs": total_pairs,
            "evaluated_pairs": len(valid_scores),
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")
    return output_data
