"""
Script containing all helper and 
plotting functions use in report
"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation 

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

# for debugging
if __name__ == '__main__':
	# df = pd.read_json('JEOPARDY_QUESTIONS1.json')
	# get_topics(df, 25)
	pass






