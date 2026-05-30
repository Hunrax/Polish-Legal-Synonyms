import numpy as np
from sklearn.cluster import DBSCAN

def cluster_synonyms(word_embeddings: dict[str, np.ndarray], eps=0.3, min_samples=2):

    words = list(word_embeddings.keys())
    matrix = np.array([word_embeddings[w] for w in words])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    
    labels = dbscan.fit_predict(matrix)

    clusters = {}
    for word, label in zip(words, labels):
        if label == -1:
            continue
        
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(word)

    return clusters