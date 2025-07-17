# Python Search Engine with TF-IDF

![Language](https://img.shields.io/badge/language-Python-3776AB?style=flat-square)
![Libraries](https://img.shields.io/badge/libs-Scikit--learn%20%7C%20NLTK%20%7C%20SpaCy-orange?style=flat-square)
![Concept](https://img.shields.io/badge/concept-NLP%20%7C%20TF--IDF-blueviolet?style=flat-square)

A "from-scratch" implementation of a search engine in Python. This project indexes a corpus of French Wikipedia articles and ranks them using TF-IDF and cosine similarity. It includes a custom TF-IDF vectorizer implementation to compare against Scikit-learn's, and benchmarks different NLP preprocessing pipelines (lemmatization vs. stemming).

## Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Technical Deep Dive](#technical-deep-dive)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [TF-IDF Vectorization](#tf-idf-vectorization)
  - [Ranking & Evaluation](#ranking--evaluation)
- [Results & Analysis](#results--analysis)
- [Future Improvements](#future-improvements)
- [License](#license)

## About The Project

This project was developed as part of an academic course on Natural Language Processing. The main objective was to build a complete information retrieval system from the ground up, covering every step from text preprocessing to performance benchmarking. The engine operates on a corpus of 2,000 French Wikipedia articles.

## Key Features

-   **TF-IDF Vectorization**: Includes both Scikit-learn's `TfidfVectorizer` and a custom, from-scratch implementation for comparison.
-   **Advanced NLP Preprocessing**: A configurable pipeline to compare the impact of **lemmatization** (with SpaCy) versus **stemming** (with NLTK).
-   **Dual-Mode Operation**:
    -   `query` mode for interactive, real-time searches via the command line.
    -   `test` mode for automated benchmarking on a predefined set of 100 queries.
-   **Cosine Similarity Ranking**: Ranks documents based on the cosine similarity between query and document vectors.
-   **Performance Benchmarking**: Calculates **Top-1** and **Top-5** accuracy to evaluate the effectiveness of different configurations.

## Built With

-   **Python 3**
-   **Scikit-learn**
-   **NLTK** (for stop-words and stemming)
-   **SpaCy** (for lemmatization)
-   **NumPy**

## Getting Started

To get a local copy up and running, follow these steps.

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/nico916/best_search_engine-.git
    ```
2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```sh
    pip install -r requirements.txt
    # You may also need to download NLTK and SpaCy data:
    # python -m nltk.downloader stopwords
    # python -m spacy download fr_core_news_sm
    ```
    *(Note: If a `requirements.txt` file is not provided, you will need to install the libraries listed under "Built With" manually.)*

## Usage

The main script `search_engine.py` can be run from the command line.

-   **To perform an interactive query:**
    ```sh
    python search_engine.py --mode query
    ```
-   **To run the benchmark with default settings (Scikit-learn + lemmatization):**
    ```sh
    python search_engine.py --mode test
    ```
-   **To test the custom vectorizer with stemming:**
    ```sh
    python search_engine.py --mode test --custom_vectorizer --preprocessing stemming
    ```

## Technical Deep Dive

The project follows a classic information retrieval pipeline.

### Preprocessing Pipeline

Each document and query goes through a normalization process:
1.  **Lowercasing**: Converts all text to lowercase.
2.  **Tokenization**: Splits text into individual words (tokens).
3.  **Stop-word Removal**: Filters out common, non-informative words.
4.  **Morphological Normalization**: Reduces words to their base form using either stemming or lemmatization.

### TF-IDF Vectorization

The core of the engine is the transformation of text into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency). I implemented a custom vectorizer to understand the inner workings and compared it to Scikit-learn's highly optimized version.

### Ranking & Evaluation

The relevance of a document to a query is determined by the **cosine similarity** between their respective TF-IDF vectors. In test mode, the engine automatically computes the Top-1 and Top-5 accuracy against a ground-truth dataset.

## Results & Analysis

| Scenario                          | Top-1 Accuracy | Top-5 Accuracy |
| --------------------------------- | :------------: | :------------: |
| A: Scikit-learn + Lemmatization   |      82%       |      97%       |
| B: Scikit-learn + Stemming        |    **85%**     |    **97%**     |
| C: Custom Vectorizer + Lemmatization |      81%       |      97%       |
| D: Custom Vectorizer + Stemming   |    **85%**     |    **97%**     |

-   **Key Insight**: Stemming consistently provided a slight edge in Top-1 accuracy over lemmatization in this context.
-   **Robustness**: All configurations achieved a high Top-5 accuracy of 97%, indicating that the correct document was almost always ranked among the most relevant results.
-   **Error Analysis**: Systematic failures on queries like `"Elizabeth Ière"` (vs. "Élisabeth Ire") highlight the model's sensitivity to exact lexical matches and the limits of a purely keyword-based approach.

## Future Improvements

-   **Semantic Search**: Integrate word embeddings (e.g., SBERT) to understand query intent beyond keywords.
-   **Query Expansion & Correction**: Add spell-checking and synonym expansion.
-   **Scalable Indexing**: Replace the in-memory matrix with a persistent inverted index (e.g., Whoosh, FAISS) for larger corpora.
-   **RAG Prototype**: Use the retrieval system as a foundation for a Retrieval-Augmented Generation (RAG) pipeline with a Large Language Model.

## License

Distributed under the MIT License. See `LICENSE` file for more information.
