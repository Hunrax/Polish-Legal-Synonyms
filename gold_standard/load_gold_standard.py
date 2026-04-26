import json
from pathlib import Path


def load_groups_from_file(file_name):
    """
    Load synonym groups from a specific orzeczenie file.
    
    Args:
        file_name (str): The file name without extension, e.g. "orzeczenie_1"
    
    Returns:
        list: List of synonym groups from that file, or empty list if file not found
    """
    json_file = Path(__file__).parent.parent / "text_extraction" / "output" / file_name / f"{file_name}_key.json"
    
    if not json_file.exists():
        print(f"Error: File '{json_file}' not found.")
        return []
    
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "groups" in data:
                print(f"Loaded {len(data['groups'])} groups from {file_name}_key.json")
                return data["groups"]
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
    
    return []


def load_lemma_mapping(file_name):
    """
    Load lemma to words mapping from lemma_mapping.txt file.
    
    Args:
        file_name (str): The file name without extension, e.g. "orzeczenie_1"
    
    Returns:
        dict: Dictionary where keys are lemmas and values are sets of words
    """
    mapping_file = Path(__file__).parent.parent / "text_extraction" / "output" / file_name / f"{file_name}_lemma_mapping.txt"
    
    lemma_mapping = {}
    
    if not mapping_file.exists():
        print(f"Error: File '{mapping_file}' not found.")
        return lemma_mapping
    
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: "lemma: [word1, word2, word3]"
                if ": [" in line:
                    lemma, words_str = line.split(": [", 1)
                    words_str = words_str.rstrip("]")
                    words = [w.strip() for w in words_str.split(",")]
                    lemma_mapping[lemma] = set(words)
    except Exception as e:
        print(f"Error reading {mapping_file}: {e}")
    
    return lemma_mapping


# not used for now
def get_word_groups_from_lemma_groups(file_name):
    """
    Convert lemma groups to word groups using lemma_mapping.
    Each lemma in a group is expanded to all its corresponding words.
    
    Args:
        file_name (str): The file name without extension, e.g. "orzeczenie_1"
    
    Returns:
        list: List of word groups (each group is a set of words)
    """
    lemma_groups = load_groups_from_file(file_name)
    lemma_mapping = load_lemma_mapping(file_name)
    
    word_groups = []
    
    for lemma_group in lemma_groups:
        word_set = set()
        
        # For each lemma in the group, add all its corresponding words
        for lemma in lemma_group:
            if lemma in lemma_mapping:
                word_set.update(lemma_mapping[lemma])
            else:
                # If lemma not found in mapping, add the lemma itself
                word_set.add(lemma)
        
        if word_set:
            word_groups.append(sorted(list(word_set)))
    
    print(f"Converted {len(lemma_groups)} lemma groups to {len(word_groups)} word groups")
    return word_groups
