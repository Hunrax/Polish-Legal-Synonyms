# Polish-Legal-Synonyms

Master's Thesis: Searching for synonyms and similar words in Polish legal texts using machine learning mechanisms.

# Comparison and Testing of Existing Solutions

We compared and tested different methods for representing word semantics in Polish, with the goal of assessing their quality in retrieving synonyms.

## Methods

### 1. **PlWordNet**

- [PlWordNet](http://plwordnet.pwr.wroc.pl/wordnet/) is a semantic lexicon for the Polish language, based on WordNet structures.
- It allows finding synonyms and semantically related words through lexical relations.
- To test it we used the [plwordnet](https://pypi.org/project/plwordnet/) library in python.

### 2. **FastText**

- FastText is an algorithm developed by Facebook AI Research for word embeddings.
- We used a pretrained FastText model for Polish available from the [fastText repository](https://fasttext.cc/docs/en/crawl-vectors.html).
- FastText uses **subword embeddings**, which makes it effective at handling word inflections and rare forms.
- The model generates vector representations of words, which were then compared to find the most semantically similar words.

### 3. **Word2Vec**

- Word2Vec is a classical method for word representation in a vector space, developed by Google.
- We used a Polish Word2Vec model of type **Continuous Skipgram**, trained on the **CoNLL 2017 corpus**.
- Model parameters: vector dimension 100, context window 10.
- The model is available via the [NLPL repository](https://vectors.nlpl.eu/repository/), and allows finding similar words based on vector embeddings.
- The Skipgram model predicts context words for a given target word, capturing semantic relations between words.

## Notes

- PlWordNet provides pretty precise lexical relations, but its coverage is unfortunately more limited compared to FastText and Word2Vec.
- Word2Vec seems to find a larger number of similar words than FastText.
