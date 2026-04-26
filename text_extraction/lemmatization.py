from pyMorfologik import Morfologik
from pyMorfologik.parsing import ListParser


def lemmatize_word(word):
    """
    Lemmatize a single word using Morfologik.
    
    Args:
        word (str): The word to lemmatize
    
    Returns:
        str: The lemma, or the original word if lemmatization fails
    """
    morf = Morfologik()
    parser = ListParser()
    
    word_lower = word.lower()
    suggestions = morf.stem([word_lower], parser)
    
    if suggestions and suggestions[0][1]:
        lemma_dict = suggestions[0][1]
        first_lemma = list(lemma_dict.keys())[0].lower()
        return first_lemma
    
    return word_lower


def lemmatize_groups(word_groups):
    """
    Convert word groups to lemma groups by lemmatizing each word.
    
    Args:
        word_groups (list): List of groups, where each group is a list/set of words
                           e.g. [['adwokata', 'adw'], ['analiza', 'ocena'], ...]
    
    Returns:
        list: List of groups with lemmatized words, duplicates removed
              e.g. [['adwokat'], ['analiza', 'ocena'], ...]
    """
    lemma_groups = []
    
    for group in word_groups:
        lemma_set = set()
        
        for word in group:
            lemma = lemmatize_word(word)
            lemma_set.add(lemma)
        
        # Convert set to sorted list for consistency
        if lemma_set:
            lemma_groups.append(sorted(list(lemma_set)))
    
    return lemma_groups
