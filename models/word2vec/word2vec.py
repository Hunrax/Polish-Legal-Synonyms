from gensim.models import KeyedVectors
import re

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "model.bin"

def group_similar_words_word2vec(words, threshold=0.7):
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    grouped = []
    seen = set()

    for w in words:
        if len(w) < 3:
            continue
        if w in seen or w not in model:
            continue

        similar = [(word, score) for word, score in model.most_similar(w, topn=100)
                   if word in words and score > threshold]

        if similar:
            group = {word for word, _ in similar}
            if len(group) > 1:
                grouped.append(sorted(group))
                seen.update(group)

    return grouped