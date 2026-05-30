import plwordnet
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "plwordnet_4_2.xml"

_wn = None

def get_wn():
    global _wn
    if _wn is None:
        print("LOADING PLWORDNET...")
        _wn = plwordnet.load(str(path))
    return _wn

def get_synonyms(wn, word):
    results = set()
    lus = wn.find(word)
    if lus is None:
        return results
    for lu in lus:
        synset = lu.synset

        for other in synset.lexical_units:
            results.add(other.name)

        for s, p, o in wn.lexical_relations_where(subject=lu):
            if "synonim" in str(p) or "przypomina" in str(p):
                results.add(o.name)
    return results


def group_similar_words_plwordnet(words):
    wn = get_wn()

    grouped = []
    seen = set()
    
    for w in words:
        if len(w) < 3:
            continue
        if w in seen:
            continue
        
        syns = get_synonyms(wn, w)
        group = (syns & words) | {w}

        group = {word for word in group if word not in seen}

        if len(group) > 1:
            grouped.append(group)
            seen.update(group)

    return grouped

def expand_via_wordnet(words, input_words):
    wn = get_wn()

    expanded_groups = []

    for w in words:
        lus = wn.find(w)
        if not lus:
            continue

        local = set()

        for lu in lus:
            synset = lu.synset

            for other in synset.lexical_units:
                if other.name in input_words:
                    local.add(other.name)

            for s, p, o in wn.lexical_relations_where(subject=lu):
                if "synonim" in str(p) or "przypomina" in str(p):
                    if o.name in input_words:
                        local.add(o.name)

        if len(local) > 1:
            expanded_groups.append(local)

    return expanded_groups