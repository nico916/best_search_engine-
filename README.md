# 🔍 Best Search Engine  
**NLP Project – Polytech ET5**

A tiny Python search engine that indexes **2 000 French-language Wikipedia articles** and ranks them with TF-IDF cosine similarity.

---

## 🚀 Quick start

```bash
# 1) Install dependencies
pip install -r requirements.txt   # std. scientific stack: numpy, scikit-learn, spacy, nltk …

# 2) Build the index + run the 100-query benchmark
python3 search_engine.py --mode test

# 3) Or switch to interactive mode
python3 search_engine.py --mode query
```

---

## 🖥️ CLI options

```bash
python3 search_engine.py -h
```

| Flag                                             | Default        | Description                                                             |
|--------------------------------------------------|---------------|-------------------------------------------------------------------------|
| `--mode {query,test}`                            | `query`       | Interactive queries or JSONL benchmark                                  |
| `--jsonl_path <file>`                            | `requetes.jsonl` | 100 queries / 50 pairs                                                  |
| `--custom_vectorizer`                            | _False_       | Use our hand-rolled TF-IDF (otherwise scikit-learn)                     |
| `--preprocessing {lemmatization,stemming}`       | `lemmatization` | Choose the text preprocessing pipeline                                  |
| `--verbosity {0,1,2}`                            | `1`           | 0 = errors only · 2 = full log                                          |

---

## 🧪 Tested scenarios

| ID | Vectorizer      | Pre-process | Top 1 Accuracy | Top 5 Accuracy |
|----|-----------------|-------------|---------------:|---------------:|
| A  | scikit-learn    | lemma       | **82 %**       | **97 %** |
| B  | scikit-learn    | stem        | **85 %**       | **97 %** |
| C  | custom          | lemma       | **81 %**       | **97 %** |
| D  | custom          | stem        | **85 %**       | **97 %** |

Example:

```bash
# scenario B : scikit-learn + stemming
python3 search_engine.py --mode test --preprocessing stemming
```

---

## 🗂️ Repository layout

```
best_search_engine/
 ├── search_engine.py        # main script
 ├── tfidf_vectorizer.py     # custom TF-IDF
 ├── wiki_split_extract_2k/  # plain-text corpus
 └── requetes.jsonl          # 100-query benchmark
```

---

## 🧠 Results at a glance

* Every scenario reaches **97 % Top 5** accuracy; three queries (“langue roumanie”, “métropolitain”, “Elizabeth Ière”) fail in every setup due to spelling or synonym issues.  
* **Stemming** fixes some false negatives (“24 heures du Mans”, “navette Columbia”) but can over-generalise.  
* The **custom vectorizer** is fully transparent but ~ 6× slower than scikit-learn and differs on a few queries (“énergie des mers”).

---

## ✨ Possible improvements

* Semantic search (SBERT / sentence embeddings)  
* Query spell-checking & correction  
* Sub-token tokenisation (`métropolitain → metro-`)  
* Persistent inverted index (Whoosh, SQLite, FAISS)  
* 🔮 Prototype **RAG** (Retrieval-Augmented Generation) to summarise answers  

---

## 📜 Licence

MIT — free for academic use and personal projects.

---

## 🤝 Contributing / bug reports

Issues welcome!

---

© 2025 – Nicolas · GitHub **@nico916**
