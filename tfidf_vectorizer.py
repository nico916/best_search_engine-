import numpy as np
import pandas as pd
from numpy.linalg import norm
from collections import defaultdict

class TFIDF_Vectorizer:
	def __init__(self):
		self.vocab = {}
		self.idf_values = {}
		self.documents = []

	def fit(self, documents):
		self.documents = documents

		# Building the vocabulary
		unique_terms = set()
		for doc in documents:
			unique_terms.update(doc)
		self.vocab = {term: idx for idx, term in enumerate(unique_terms)}

		# Calculate document frequency for each term
		doc_freq = defaultdict(int)
		for term in unique_terms:
			doc_freq[term] = sum(1 for doc in documents if term in doc)

		# Calculate IDF for each term
		num_docs = len(documents)
		self.idf_values = {term: np.log((num_docs+1) / (doc_freq[term]+1)) for term in unique_terms}
		self.idf_ = list(self.idf_values.values())

	def transform(self, documents):
		# Initialize the TF-IDF matrix
		tf_idf_matrix = np.zeros((len(documents), len(self.vocab)))

		for doc_idx, doc in enumerate(documents):
			# Count the number of occurences of the terms in a document
			term_counts = defaultdict(int)
			for token in doc:
				term_counts[token] += 1

			# Set the TF-IDF values for a document
			for token, count in term_counts.items():
				if token in self.vocab:
					tf = count / len(doc)
					idf = self.idf_values[token]
					tf_idf_matrix[doc_idx,self.vocab[token]] = tf * idf

		row_norms = norm(tf_idf_matrix, axis=1, ord=2, keepdims=True)
		row_norms[row_norms == 0] = 1  # Replace zero norms with 1 to avoid division by zero
		tf_idf_matrix = tf_idf_matrix / row_norms
		return tf_idf_matrix

	def fit_transform(self, documents):
		self.fit(documents)
		return self.transform(documents)

	def get_feature_names_out(self):
		return list(self.vocab.keys())