"""
Script containing all helper and 
plotting functions use in report
"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import nltk
import re

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.similarities.annoy import AnnoyIndexer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation 


# preload dataframe
df = pd.read_json("JEOPARDY_QUESTIONS1.json")

def get_topics(df, num_topics, rd:str=None, max_df:float=0.95):
	"""
	Finds all topics for the 'question'
	column of data for the specified round.
	Arguments:
		- df: DataFrame of jeopardy data
		- rd: Specifies the round to subset
				 on. Should be one of 'Jeopardy!',
				 'Double Jeopardy!',
				 'Final Jeopardy!' or None. If
				 None, use entire data
		- max_data: Passed as 'max_data' in the 
					 TfidataVectorizer. Specifies a
					 cutoff point to exclude terms
					 with too high document
					 frequency.
	Returns:
		W: np.array, the document-topic matrix
		H: np.array, the topic-term matrix
		data: pd.DataFrame, df with new column 'topic'
			 specifying the topic for the question
		vocab: dict, word to index from TFIDF
	"""
	if rd is not None:
		data = df.loc[df['round'] == rd]
	else:
		data = df.copy()

	# data prep
	data.fillna(0)  # for missing question values
	# convert value from $500 -> 500.0
	data.loc[:, 'value'] = data['value'].str[1:].str.replace(',','').astype(np.float32)
	data.loc[:, 'air_date'] = pd.to_datetime(data.loc[:, 'air_date'])
	data.loc[:, 'year'] = data.loc[:, 'air_date'].dt.year

	# tfidf
	vectorizer = TfidfVectorizer(
		strip_accents='ascii',
		stop_words='english',
		max_df=max_df
	)
	tfidf_matrix = vectorizer.fit_transform(data['question'])

	# NMF
	model = NMF(n_components=num_topics, init='random', max_iter=400)
	W = model.fit_transform(tfidf_matrix)
	H = model.components_

	topics = np.apply_along_axis(np.argmax, axis=1, arr=W)
	data.loc[:, 'topic'] = topics

	return W, H, data, vectorizer.vocabulary_

def get_difficulty(df, rd: str):
	"""
	Get difficulty for each value group.
	"""
	pass

def w2v(df):
    categories = set([i.lower() for i in df['category']])
    categories = list(categories)
    categories = [re.sub('[^a-zA-Z]', ' ', cat) for cat in categories]
    categories = [re.sub(r'\s+', ' ', cat) for cat in categories]
    categories = [c.strip() for c in categories]
    category = [nltk.sent_tokenize(c) for c in categories]
    all_words = [nltk.word_tokenize(c) for cat in category for c in cat]
    for i in range(len(all_words)):
        all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]
    word2vec = Word2Vec(all_words, min_count=2)
    vocabulary = word2vec.wv
    return word2vec, vocabulary

def summary(x:str, word2vec, vocabulary):
    annoy_index = AnnoyIndexer(word2vec, 100)
    vector = vocabulary[x]
    # The instance of AnnoyIndexer we just created is passed
    approximate_neighbors = vocabulary.most_similar([vector], topn=11, indexer=annoy_index)
    # Neatly print the approximate_neighbors and their corresponding cosine similarity values
    print("Approximate Neighbors")
    for neighbor in approximate_neighbors:
        print(neighbor)
    normal_neighbors = vocabulary.most_similar([vector], topn=11)
    print("\nExact Neighbors")
    for neighbor in normal_neighbors:
        print(neighbor)


# for debugging
if __name__ == '__main__':
	# df = pd.read_json('JEOPARDY_QUESTIONS1.json')
	# get_topics(df, 25)
	pass






