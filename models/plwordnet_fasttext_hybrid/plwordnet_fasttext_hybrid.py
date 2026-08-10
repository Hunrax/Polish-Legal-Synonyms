from collections import defaultdict
import sys
from pathlib import Path

import spacy

# Add the workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.plwordnet.plwordnet import group_similar_words_plwordnet, expand_via_wordnet
from models.fasttext.fasttext import group_similar_words_fasttext

nlp = spacy.load("pl_core_news_lg")

ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def get_word_pos(words):
    pos_map = {}

    docs = nlp.pipe(words)

    for word, doc in zip(words, docs):
        if not doc:
            continue

        token = doc[0]

        if token.pos_ in ALLOWED_POS:
            pos_map[word] = token.pos_
        else:
            pos_map[word] = None

    return pos_map


def split_group_by_pos(group, pos_map):
    pos_groups = defaultdict(set)

    for word in group:
        pos = pos_map.get(word)

        if pos is None:
            continue

        pos_groups[pos].add(word)

    valid_groups = [
        g for g in pos_groups.values()
        if len(g) > 1
    ]

    return valid_groups


def normalize_groups_by_pos(groups):
    all_words = set()

    for g in groups:
        all_words.update(g)

    pos_map = get_word_pos(all_words)

    normalized = []

    for group in groups:
        split_groups = split_group_by_pos(group, pos_map)
        normalized.extend(split_groups)

    return normalized


def merge_groups(groups):
    merged = []

    for g in groups:
        g = set(g)
        placed = False

        for mg in merged:
            if mg & g:
                mg |= g
                placed = True
                break

        if not placed:
            merged.append(g)

    changed = True
    while changed:
        changed = False
        new_merged = []

        for g in merged:
            placed = False
            for ng in new_merged:
                if ng & g:
                    ng |= g
                    placed = True
                    changed = True
                    break
            if not placed:
                new_merged.append(g)

        merged = new_merged

    return merged

def group_similar_words_hybrid(words, threshold):
    fasttext_groups = group_similar_words_fasttext(words, threshold)

    for i, group in enumerate(fasttext_groups, start=1):
        print(f"FastText Group {i}: {', '.join(group)}")

    fasttext_groups = normalize_groups_by_pos(fasttext_groups)

    #TODO: Uncomment when PLWordNet is available
    # for i, group in enumerate(fasttext_groups, start=1):
    #     print(f"FastText Group Normalized {i}: {', '.join(group)}")
    # plwordnet_groups = group_similar_words_plwordnet(set(words))

    # for i, group in enumerate(plwordnet_groups, start=1):
    #     print(f"PLWordNet Group {i}: {', '.join(group)}")
    
    # plwordnet_groups = normalize_groups_by_pos(plwordnet_groups)
    # for i, group in enumerate(plwordnet_groups, start=1):
    #     print(f"PLWordNet Group Normalized {i}: {', '.join(group)}")

    # ft_words = set().union(*fasttext_groups) if fasttext_groups else set()
    # wn_expanded = expand_via_wordnet(ft_words, words)
    # for i, group in enumerate(wn_expanded, start=1):
    #     print(f"WordNet Expanded Group {i}: {', '.join(group)}")

    all_groups = fasttext_groups
    # all_groups = plwordnet_groups + fasttext_groups + wn_expanded

    merged = merge_groups(all_groups)
    for i, group in enumerate(merged, start=1):
        print(f"Merged Group (Final) {i}: {', '.join(group)}")

    merged = normalize_groups_by_pos(merged)
    for i, group in enumerate(merged, start=1):
        print(f"Merged Group Normalized (Final) {i}: {', '.join(group)}")

    return merged