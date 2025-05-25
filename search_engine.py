import argparse
import os
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description="""The Best Search Engine™: Research a french wikipedia article, the closest within our database will be outputed.""")
parser.add_argument(
        '--mode', 
        choices=['query', 'test'], 
        required=True,
        help="Choose 'query' to manually enter queries or 'test' to test performance on a JSONL file."
    )
parser.add_argument(
        '--jsonl_path', 
        type=str, 
        default='requetes.jsonl', 
        help="""Path to the queries JSONL file. Defaults to 'requetes.jsonl'. 
        	    The JSONL file indicated here should have the same structure as the default one."""
    )
parser.add_argument(
        '--custom_vectorizer', 
        action='store_true', 
        help="Use custom TF-IDF vectorizer implementation. If not set, scikit-learn's implementation will be used."
    )
parser.add_argument(
        '--preprocessing', 
        choices=['lemmatization', 'stemming'], 
        required=True, 
        help="Select the preprocessing method: 'lemmatization' or 'stemming'."
    )
parser.add_argument(
        '--verbosity', 
        type=int, 
        choices=[0, 1, 2], 
        default=1, 
        help="Set the verbosity level of results: 0 (minimal), 1 (default), 2 (detailed)."
    )


class SearchEngine:
	def __init__(self, documents_path, jsonl_path, mode, custom_vectorizer, preprocessing, verbosity):
		self.documents_path = documents_path
		self.jsonl_path = jsonl_path
		self.mode = mode
		self.custom_vectorizer = custom_vectorizer
		self.preprocessing = preprocessing
		self.verbosity = verbosity
		self.documents = []
		self.document_names = []
		self.stopwords = set(stopwords.words('french'))
		self.tfidf_matrix = None
		if custom_vectorizer:
			from tfidf_vectorizer import TFIDF_Vectorizer
			self.vectorizer = TFIDF_Vectorizer()
		else :
			from sklearn.feature_extraction.text import TfidfVectorizer
			self.vectorizer = TfidfVectorizer()
		if preprocessing == 'lemmatization':
			import spacy
			self.lemmatizer = spacy.load('fr_core_news_sm')
		else :
			from nltk.stem.snowball import SnowballStemmer
			self.stemmer = SnowballStemmer("french")

	def preprocess(self, document):
		document = document.lower()
		if self.preprocessing == 'lemmatization':
			doc = self.lemmatizer(document)
			preprocessed_document = [
				token.lemma_ for token in doc 
				if token.lemma_ not in self.stopwords and not token.is_punct
			]
		else :
			import re
			pattern = r"\b\w[\w\-]+\b"
			doc = re.findall(pattern, document)
			preprocessed_document = [
				self.stemmer.stem(token) for token in doc
				if token not in self.stopwords
			]
		if not self.custom_vectorizer:
			preprocessed_document = ' '.join(preprocessed_document)
		return preprocessed_document

	def create_vector_space(self):
		for root, _, files in os.walk(self.documents_path):
			self.root = root
			for file_name in files:
				file_path = os.path.join(root, file_name)
				with open(file_path, 'r', encoding='utf-8') as file:
					content = file.read()
					preprocessed_document = self.preprocess(content)
					self.documents.append(preprocessed_document)
					self.document_names.append(file_name)
		self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

	def search(self, query):
		preprocessed_query = self.preprocess(query)
		query_vector = self.vectorizer.transform([preprocessed_query])
		similarities = cosine_similarity(query_vector, self.tfidf_matrix)
		similarity_scores = similarities.flatten()
		
		if self.mode == 'query':
			best_index = np.argmax(similarity_scores)
			print("The most relevant article to your query is: ", self.document_names[best_index])
			with open(os.path.join(self.root, self.document_names[best_index])) as file:
				content = file.read()
				print(content)
			print()
		else :
			ranked_indices = np.argsort(similarity_scores)[::-1]
			ranked_results = [
				(self.document_names[i], similarity_scores[i])
				for i in ranked_indices
			]
			return ranked_results

	def test_queries(self):
		queries_df = pd.read_json(path_or_buf=self.jsonl_path, lines=True, encoding='utf-8')
		test_queries = [query for queries in queries_df["Queries"] for query in queries]
		test_answer = [doc for doc, queries in zip(queries_df["Input file"], queries_df["Queries"]) for _ in queries]
		correct_matches = 0
		matches_in_top5 = 0
		if self.verbosity == 2: previous_failed = False

		if len(test_queries) != len(test_answer):
			print("ERROR: The number of queries and input files doesn't match")

		for idx in range(len(test_queries)):
			query = test_queries[idx]
			expected_file = test_answer[idx]
			ranked_results = self.search(query)
			index_expected_file = next((i for i,(doc_name,_) in enumerate(ranked_results) if doc_name == expected_file),-1)
			if self.verbosity == 0:
				if index_expected_file == 0:
					correct_matches += 1
					matches_in_top5 += 1
				else :
					if index_expected_file < 5 and index_expected_file != -1:
						print(f"Failed match for query: {query} but the expected article is in the 5 most relevant articles")
						print()
						matches_in_top5 += 1
					else :
						print(f"Failed match for query: {query}")
						print()
			elif self.verbosity == 1:
				if index_expected_file == 0:
					correct_matches += 1
					matches_in_top5 += 1
				else :
					if index_expected_file < 5 and index_expected_file != -1:
						matches_in_top5 += 1
					print(f"Failed match for query: {query}.")
					print(f"Expected file: {expected_file}. Index of the expected file: {index_expected_file}")
					print("The 5 most relevant articles found are:")
					for i in range(5):
						print(f"Article {ranked_results[i][0]} with similarity {ranked_results[i][1]}")
					print()
			else :
				if index_expected_file == 0:
					correct_matches += 1
					matches_in_top5 += 1
					previous_failed = False
					print("Article {expected_file} correctly retrieve")
				else :
					if not previous_failed : print()
					previous_failed = True
					if index_expected_file < 5 and index_expected_file != -1:
						matches_in_top5 += 1
					print(f"Failed match for article query: {query}.")
					print(f"Expected file: {expected_file}. Index of the expected file: {index_expected_file}")
					print("The 5 most relevant articles found are:")
					for i in range(5):
						print(f"Article {ranked_results[i][0]} with similarity {ranked_results[i][1]}")
					print()
		accuracy = (correct_matches / len(test_queries)) * 100
		accuracy_top5 = (matches_in_top5 / len(test_queries)) * 100
		print(f"Accuracy : {accuracy:.2f}%")
		print(f"Accuracy in 5 most relevant articles : {accuracy_top5:.2f}%")


if __name__ == "__main__":
	args = parser.parse_args()
	directory = 'wiki_split_extract_2k'
	engine = SearchEngine(directory, args.jsonl_path, args.mode, args.custom_vectorizer, args.preprocessing, args.verbosity)
	engine.create_vector_space()
	if args.mode == 'query':
		while(True):
			print("Please enter your query. If you which to exit, please type 'x'.")
			query = input()
			if (query.lower() == 'x'):
				break
			engine.search(query)
	else :
		engine.test_queries()
