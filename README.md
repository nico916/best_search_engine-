# best_search_engine

The main script of the search engine is search_engine.py. The script tfidf_vectorizer.py contains our own implementation of a TF-IDF Vectorizer. Our search engine also uses scikit-learn's TF-IDF Vectorizer.

The command `python3 search_engine.py -h` prints the different options that can be used to launch the search engine:

```
The Best Search Engine™: Research a french wikipedia article, the closest within our database will be outputed.

options:
  -h, --help            show this help message and exit
  --mode {query,test}   Choose 'query' to manually enter queries or 'test' to test performance on a JSONL file.
  --jsonl_path JSONL_PATH
                        Path to the queries JSONL file. Defaults to 'requetes.jsonl'. The JSONL file indicated here should have the same structure as the
                        default one.
  --custom_vectorizer   Use custom TF-IDF vectorizer implementation. If not set, scikit-learn's implementation will be used.
  --preprocessing {lemmatization,stemming}
                        Select the preprocessing method: 'lemmatization' or 'stemming'.
  --verbosity {0,1,2}   Set the verbosity level of results: 0 (minimal), 1 (default), 2 (detailed).
```

The following commands correspond to the four scenarios described in our project report:

- Scenario A: `python3 search_engine.py --mode test --preprocessing lemmatization`

- Scenario B: `python3 search_engine.py --mode test --preprocessing stemming`

- Scenario C: `python3 search_engine.py --mode test --custom_vectorizer --preprocessing lemmatization`

- Scenario D: `python3 search_engine.py --mode test --custom_vectorizer --preprocessing stemming`