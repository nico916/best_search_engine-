# 🔍 Best Search Engine  
**Projet NLP – Polytech ET5**  

Petit moteur de recherche en Python qui indexe **2 000 articles Wikipédia (fr)** et les classe par similarité TF-IDF + cosinus.

<p align="center">
  <img src="docs/pipeline_diagram.png" alt="Pipeline du projet" width="500">
</p>

---

## 🚀 Quick start

```bash
# 1) Installer les dépendances
pip install -r requirements.txt

# 2) Construire l’index + lancer un benchmark 100 requêtes
python3 search_engine.py --mode test

# 3) Ou passer en mode interactif
python3 search_engine.py --mode query
```

---

## 🖥️ CLI options

```bash
python3 search_engine.py -h
```

| Flag                          | Valeur par défaut | Description                                                    |
|-------------------------------|-------------------|----------------------------------------------------------------|
| `--mode {query,test}`         | `query`           | Requêtes interactives ou benchmark JSONL                       |
| `--jsonl_path <fichier>`      | `requetes.jsonl`  | Fichier de 100 requêtes / 50 paires                            |
| `--custom_vectorizer`         | _False_           | Active notre TF-IDF maison (sinon scikit-learn)                |
| `--preprocessing {lemmatization,stemming}` | `lemmatization` | Choix du pré-traitement                                       |
| `--verbosity {0,1,2}`         | `1`               | 0 = seulement les erreurs, 2 = log complet                     |

---

## 🧪 Scénarios testés

| ID | Vectorizer     | Pré-process | Accuracy Top 1 | Accuracy Top 5 |
|----|----------------|-------------|---------------:|---------------:|
| A  | scikit-learn   | lemma       | **82 %**       | **97 %** |
| B  | scikit-learn   | stem        | **85 %**       | **97 %** |
| C  | custom         | lemma       | **81 %**       | **97 %** |
| D  | custom         | stem        | **85 %**       | **97 %** |

Exécuter un scénario :

```bash
# scénario B : scikit-learn + stemming
python3 search_engine.py --mode test --preprocessing stemming
```

---

## 🗂️ Structure du dépôt

```
best_search_engine/
 ├── search_engine.py          # point d’entrée
 ├── tfidf_vectorizer.py       # TF-IDF maison
 ├── wiki_split_extract_2k/    # corpus txt
 ├── requetes.jsonl            # 100 requêtes benchmark
 ├── requirements.txt
 └── docs/
     └── pipeline_diagram.png
```

---

## 🧠 Analyse rapide des résultats

* Tous les scénarios atteignent **97 % Top 5** ; trois requêtes (« langue roumanie », « métropolitain », « Elizabeth Ière ») échouent systématiquement à cause de variantes lexicales / orthographiques.  
* **Stemming** règle certains faux négatifs (« 24 heures du Mans », « navette Columbia ») mais introduit de la sur-généralisation.  
* Le **vectorizer custom** est transparent mais ~6× plus lent que scikit-learn et diverge sur quelques requêtes (« énergie des mers »).  

---

## ✨ Améliorations possibles

* Recherche sémantique (SBERT / embeddings)
* Vérification/correction orthographique des requêtes
* Tokenisation sous-mot (`métropolitain → metro-`)
* Index inversé persistant (Whoosh, SQLite, FAISS)
* 🔮 Prototype **RAG** (Retrieval-Augmented Generation) pour résumer la réponse

---

## 📜 License

MIT – libre pour usage académique et projets personnels.  

---

## 🤝 Contribuer / Signaler un bug

Issues et Pull Requests bienvenus !  

---

© 2025 – Nicolas *pseudo GitHub @nico916*
