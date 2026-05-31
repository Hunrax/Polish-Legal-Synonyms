# Polish-Legal-Synonyms

Master's Thesis: Searching for synonyms and similar words in Polish legal texts using machine learning mechanisms.

# Project Overview
Polish legal texts (such as legislative acts, regulations, and court rulings) utilize highly formalized language, heavily relying on specific chains of synonyms and near-synonyms (e.g., podsądny - oskarżony - pozwany). The goal of this project was to thoroughly analyze and test various machine learning mechanisms for extracting synonyms, and to develop an improved hybrid solution optimized for the specific characteristics of legal documents.

## Architecture and Main Project Sections
The project pipeline is divided into several key analytical modules:

### 1. Data Extraction & Preprocessing
  * Text streams are extracted directly from raw legal documents (PDF/DOCX).
  * For vector models and PlWordNet, the text undergoes deep cleaning and lemmatization (reducing words to their base forms) using the spaCy and pyMorfologik libraries.
  * For contextual models (BERT), words are extracted without lemmatization to preserve their natural, original sentence context.
### 2. Comparison & Clustering
  * Static models (vector-based): Group words based on cosine similarity and a defined distance threshold.
  * Deep models (BERT): Utilize the DBSCAN algorithm to cluster contextual vectors based on spatial density (using the epsilon parameter).
### 3. Validation (LLM-as-a-Judge)
  * Due to the lack of a "gold standard" reference dataset for Polish legal texts, the system automatically verifies the quality of word pairs using a Large Language Model: speakleash/Bielik-11B-v2.2-Instruct.
  * The LLM judge scores each extracted word pair on a scale of 1-5. Based on these scores, the system calculates custom metrics: Mean Score, Acceptance Rate, and Coverage.
### 4. Proposed Hybrid Solution
  * The final custom script (plwordnet_fasttext_hybrid.py) synergistically combines the dictionary-based precision of PlWordNet with the broad coverage of the FastText model.
  * Key enhancements include: filtering out vector noise (excluding months and time periods), cross-expanding clusters via WordNet synsets (expand_via_wordnet), and strict morphological filtering using spaCy (rejecting clusters that mix different Parts of Speech).

## Tested Models and Methods (Existing Solutions)
During the research phase, the following architectures were evaluated for their ability to capture semantic representations in the Polish language:

### 1. PlWordNet (Słowosieć)
* A lexical-semantic database for the Polish language based on WordNet structures.
* Utilizes expert-defined direct relations (synsets). It achieves the highest precision and acceptance rate, but due to its closed dictionary nature, its overall text coverage is relatively low.
### 2. FastText
* A word embedding algorithm developed by Facebook AI Research.
* Utilizes character n-grams (subword embeddings), making it highly effective at handling Polish inflection and rare words. It provides excellent text coverage but tends to group associative words (thematic connections) rather than strict synonyms.
### 3. Word2Vec
* A classic method for static word representation. We used a Polish Continuous Skipgram model (100-dimensional vectors from the NLPL repository).
* While it finds more related words than FastText, it generally yields lower overall quality and creates more inaccurate connections.
### 4. Contextual Models (BERT variants)
We evaluated deep transformer architectures that generate dynamic embeddings based on the surrounding sentence context. Three variants were tested:
* dkleczek/bert-base-polish-cased-v1 (PolBERT)
* allegro/herbert-large-cased (HerBERT)
* nlpaueb/legal-bert-base-uncased (Legal-BERT - domain-specific model)

## Final Results
Based on the analysis of the baseline results, a custom hybrid solution was implemented. It integrates PlWordNet with FastText algorithm (using a high threshold of 0.7-0.8) and enforces Part-of-Speech consistency. This approach successfully eliminated "vector noise", maximized the mean score of the accepted pairs, and maintained a satisfactory coverage level of the legal source texts.

## Authors
* Jan Barczewski
* Radosław Gajewski
