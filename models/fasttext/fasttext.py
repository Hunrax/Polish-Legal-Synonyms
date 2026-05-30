from pathlib import Path
from gensim.models.fasttext import load_facebook_vectors

path = "cc.pl.300.bin"

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "cc.pl.300.bin"

_model = None

def get_model():
    global _model
    if _model is None:
        print("LOADING FASTTEXT MODEL...")
        _model = load_facebook_vectors(model_path)
    return _model

def group_similar_words_fasttext(words, threshold):
    model = get_model()

    grouped = []
    seen = set()

    for w in words:
        if len(w) < 3:
            continue
        if w in seen or w not in model:
            continue
        
        similar = [
            (word, score)
            for word, score in model.most_similar(w, topn=100)
            if word in words and score > threshold
        ]

        if similar:
            group = {word for word, _ in similar}

            if len(group) > 1:
                grouped.append(group)
                seen.update(group)

    return grouped