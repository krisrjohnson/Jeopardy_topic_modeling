"""
Script containing all helper and 
plotting functions use in report
"""
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import nltk
import re
import time
import string
import gensim
import spacy
from collections import Counter, defaultdict
from wordcloud import WordCloud
from collections import Counter

from nltk.corpus import stopwords
from gensim.models import Word2Vec, ldamodel
from gensim.similarities.annoy import AnnoyIndexer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import jaccard_score


# preload dataframe
df = pd.read_json("JEOPARDY_QUESTIONS1.json")


def get_topics(df, num_topics, rd: str = None, max_df: float = 0.95, min_df: int = 3, ret_tfidf=False):
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
       - min_df: TfidataVectorizer min freq count
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

    # tfidf
    vectorizer = TfidfVectorizer(
        strip_accents='ascii',
        stop_words='english',
        max_df=max_df,
        min_df=min_df
    )
    tfidf_matrix = vectorizer.fit_transform(data['question'])

    # NMF
    model = NMF(n_components=num_topics, init='random', max_iter=400)
    W = model.fit_transform(tfidf_matrix)
    H = model.components_

    topics = np.argmax(W, axis=1)
    data.loc[:, 'topic'] = topics

    if ret_tfidf:
        return W, H, data, vectorizer.vocabulary_, tfidf_matrix

    return W, H, data, vectorizer.vocabulary_

def get_difficulty(df):
    """
    Get difficulty for each value group.
    Arguments:
      - df: jeopardy question data
    Returns:
      - data: pd.DataFrame, df with a 
              difficulty column
    """
    data = df.copy()
    data.loc[:, 'difficulty'] = data.groupby(
        ['show_number']).rank(method='dense')['value']
    return data


def get_relevancies(df, num_topics=25, top_topics=3, col='difficulty', verbose=True, max_df=0.95, min_df=3):
    """
    Given df with column, find the top
    'num_topics' topics for each unique column value and
    the relevant terms in those topics.
    Arguments:
      - df: DataFrame, must have column specified, defaults to 'difficulty' 
            column
      - num_topics: the number of top most common
                    topics to look at
    Returns:
      - results: dict, attributes:
        - 'W': document-topic matrix
        - 'H': topic-term matrix
        - 'data': df with column sepcified and topic columns
        - 'relevant_words': sorted word relevancy
        - 'top3': word relevancies for the top3 most
                  most common topics
    """
    results = {}

    if col == 'difficulty':
        groups = range(1,7)
    else:
        groups = df[col].unique()

    for col_val in groups:
        if verbose:
            print(f'col: {col}, val: {col_val}: Computing topics...')
        W, H, data, vocab = get_topics(df.loc[df[col] == col_val], num_topics, max_df=max_df, min_df=min_df)
        idx2word = {idx: word for word, idx in vocab.items()}

        if verbose:
            print(f'\t{col_val}: Finding word relevancies...')
        sorted_term_relevancy = np.argsort(-H, axis=1)
        word_relevancy = [list(map(idx2word.get, topic))
                          for topic in sorted_term_relevancy]

        if verbose:
            print(f'\t{col_val}: Finding top {num_topics} most common topics...')
        topics = np.argsort(
            # 'value' is arbitrary column
            data.groupby('topic').count()['value']
        )

        top = []
        max_length = min(num_topics, top_topics)
        for topic in topics[:max_length]:
            top.append(word_relevancy[topic])

        results[col_val] = {
            'W': W,
            'H': H,
            'data': data,
            'relevant_words': word_relevancy,
            'top': top
        }

    return results

def prep_data(df):
    "Function to prep data"
    data = df.copy()

    data = data.fillna(0)  # for missing question values
    # convert value from $500 -> 500.0
    data.loc[:, 'value'] = data['value'].str[1:].str.replace(',', '').astype(np.float32)
    data.loc[:, 'air_date'] = pd.to_datetime(data.loc[:, 'air_date'])
    data.loc[:, 'year'] = data.loc[:, 'air_date'].dt.year
    data.question = data.question.apply(lambda x: clean_text(x))

    return data


def w2v(df):
    '''Convert categories into Word2Vecs'''
    categories = set([i.lower() for i in df['category']])
    categories = list(categories)
    categories = [re.sub('[^a-zA-Z]', ' ', cat) for cat in categories]
    categories = [re.sub(r'\s+', ' ', cat) for cat in categories]
    categories = [c.strip() for c in categories]
    category = [nltk.sent_tokenize(c) for c in categories]
    all_words = [nltk.word_tokenize(c) for cat in category for c in cat]
    for i in range(len(all_words)):
        all_words[i] = [w for w in all_words[i]
                        if w not in stopwords.words('english')]
    word2vec = Word2Vec(all_words, min_count=2)
    vocabulary = word2vec.wv
    return word2vec, vocabulary


def summary(x: str, word2vec, vocabulary):
    '''Summary of the 10 nearest similar words based on Word2Vec'''
    annoy_index = AnnoyIndexer(word2vec, 100)
    vector = vocabulary[x]
    # The instance of AnnoyIndexer we just created is passed
    approximate_neighbors = vocabulary.most_similar(
        [vector], topn=11, indexer=annoy_index)
    # Neatly print the approximate_neighbors and their corresponding cosine similarity values
    print("Approximate Neighbors")
    for neighbor in approximate_neighbors:
        print(neighbor)
    normal_neighbors = vocabulary.most_similar([vector], topn=11)
    print("\nExact Neighbors")
    for neighbor in normal_neighbors:
        print(neighbor)

# adapted from:
# https://towardsdatascience.com/topic-modeling-quora-questions-with-lda-nmf-aff8dce5e1dd        
def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


def tokenize(questions:list):
    for question in questions:
        yield(gensim.utils.simple_preprocess(str(question)))
        
        
def lemmatize(text, word_type=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    out_text = []
    for sentence in text:
        doc = nlp(" ".join(sentence)) 
        out_text.append([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in word_type and token.text not in stopwords.words('english')])
    return out_text


def lda_topics(questions, topics:int):
    id2word = gensim.corpora.Dictionary(questions)
    corpus = [id2word.doc2bow(text) for text in questions]
    lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=topics)
    return id2word, corpus, lda
        
    
def topic_df(lda, topics:int):
    words_dict = {};
    for i in range(topics):
        words = lda.show_topic(i, topn = 15);
        words_dict[f'Topic {(i+1):2d}'] = [i[0] for i in words];
    return pd.DataFrame(words_dict)


def most_likely(corpus, lda):
    most_likely_topic=[]
    for doc in corpus:
        likelihoods = dict(lda.get_document_topics(doc))
        most_likely_topic.append(max(likelihoods,key=likelihoods.get))
    return most_likely_topic


def top_ic(df):
    dfs = [x for _, x in df.groupby('round')]
    num_topics = len(df['topic'].unique())
    top_topics=[]
    for df in dfs[:-1]:
        topic_counts = [0]*num_topics
        for topic in df['topic']:
            topic_counts[topic]+=1
        topics_perc = [count/sum(topic_counts) for count in topic_counts]
        top_topics.append((-np.array(topics_perc)).argsort()[:5])
    return top_topics
    

def plot_distributions(df):
    year_idx = []
    rows = 7
    cols = 4
    years = {i:[0]*20 for i in range(1984,2013)}
    for i, row in df.iterrows():
        topic = years[row['year']]
        topic[row['topic']]+=1
        years[row['year']] = topic
    for year in years:
        years[year] = [count/sum(years[year]) for count in years[year]]
        idx = (-np.array(years[year])).argsort()[:5]
        year_idx.append(idx)
    fig, axs = plt.subplots(7,4,figsize=(22,12))
    index= 0
    years = list(d.keys())
    for r in range(rows):
        for c in range(cols):
            axs[r,c].bar(range(20), years[years[index]])
            axs[r,c].set_title(years[index])
            axs[r,c].set_ylim([0.0, 0.14])
            index +=1
    return year_idx
    
def get_topic_word_counts(df, vocab, tfidf_matrix, verbose=False):
    '''
    # Given original dataframe with topics column, vocab, and sparse matrix
    # where rows of sparse_matrix align with rows of dataframe
    # create defaultdict whose keys are the topics
    # and values are words with counts
    '''
    idx2word = {idx: word for word, idx in vocab.items()}
    vf = np.vectorize((lambda x: idx2word[x]))

    start = time.time()
    topic_counts = defaultdict(Counter)

    checkpoints = [20_000 * _ for _ in range(tfidf_matrix.shape[0])]

    for i in range(tfidf_matrix.shape[0]):
        _, cols = tfidf_matrix[i, :].nonzero()
        if len(cols) == 0:
            continue
        topic_counts[df.loc[i].topic].update(set(vf(cols)))
        #     print(set(vf(cols)))
        if verbose:
            if i in checkpoints:
                print(f'{i:,}, {time.time() - start}')

    return topic_counts


def generate_wc(word_freqs):
    '''given a dictionary of words and their counts, generate and plot a wordcloud'''
    wc = WordCloud(width=600, height=600,
                   background_color='white',
                   # stopwords=set(STOPWORDS),
                   min_font_size=10)
    wc.generate_from_frequencies(word_freqs)
    return wc


def compute_similarities(J, D, max_elements, normalize=True):
    """
    Computes the mean_max_Jaccard similarity between
    every group in J to every element in D. 
    Arguments:
      - J: First set, (J is for Jeopardy!)
      - D: Second set, (D is for Double Jeopardy!)
      - max_elements: passed to mean_max_Jaccard
    Returns:
      - sim_matrix: np.array[np.array] which is 
                    |J| x |D| and sim_matrix[i, j]
                    is the mean_max_Jaccard
                    between J[i] and D[j]
    """
    sim_matrix = np.zeros([len(J), len(D)])

    for j, set_j in enumerate(J):
        for d, set_d in enumerate(D):
            sim_matrix[j, d] = mean_max_Jaccard(
                set_j, set_d, max_elements
            )

    if normalize:
        # min-max scaling
        sim_matrix = (sim_matrix - sim_matrix.min()) / \
            (sim_matrix.max() - sim_matrix.min())
    sim_matrix = np.round(sim_matrix, 4)
    Jaccard_df = pd.DataFrame(sim_matrix, columns=range(
        1, len(D)+1), index=range(1, len(J)+1))

    return Jaccard_df


def mean_max_Jaccard(A, B, max_elements=None) -> int:
    """
    Computes the mean of the max Jaccard
    similarities between each set in A and any other
    set in B. In this case, A and B are made up of many
    subsets, for example, A and B are lists of lists 
    where each sublist is a sorted set of relevant terms
    in a topic where A is $200 questions in Jeopardy!
    round and B is $400 questions in Double Jeopardy.
    Arguments:
      - A: An iterable where elements are iterables 
      - B: Same for A
      - max_elements: The maximum number of elements
                      to consider
    Returns:
      - similarity: int, single value representing the
                    mean similarity scores of the max
                    similarity between each topic and A
                    and any topic in B.
    """
    topic_scores = []
    for a in A:
        jaccard_a_B = []  # the ith element is jaccard_score between a and Bi
        for b in B:
            # the num of words is different for each topic
            min_set_size = min(len(a), len(b))
            if max_elements is not None:
                max_elements = min(min_set_size, max_elements)

            jaccard_a_B.append(jaccard_score(
              a[:max_elements], b[:max_elements], average='macro'))
        # log for comparison, prevent log(0)
        topic_scores.append(np.log(np.max(jaccard_a_B)+1e-8))

    return np.mean(topic_scores)

def plot_Jaccard_matrix(Jaccard_matrix, threshold=0.7):
    """
    Function to plot the strength of connections
    between the groups represented by the 
    Jaccard matrix.

    """
    fig, ax = plt.subplots(figsize=(8, 10))

    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(.5, 6.8)

    ax.scatter(np.zeros(6), range(1,7), s=500, c='red', label='Jeopardy!', zorder=10)
    ax.scatter(np.ones(6), range(1,7), s=500, c='blue', label='Double Jeopardy', zorder=10)
    ax.legend()

    for J_difficulty in Jaccard_matrix.columns:
        for DBL_difficulty in Jaccard_matrix.index:
            if Jaccard_matrix.loc[J_difficulty, DBL_difficulty] > threshold:
                ax.plot(
                    [0, 1], [J_difficulty, DBL_difficulty],
                    lw=8, alpha=0.7,
                    c='black'
                )

    ax.set_title("Similarity between Jeopardy! and Double Jeopardy! topics", fontsize=18)
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    ax.set_xticks([0,1])
    ax.set_yticks([1,2,3,4,5,6])
    ax.tick_params('both', width=0, labelsize=18)
    ax.set_xlabel('Round', fontsize=15)
    ax.set_ylabel('Difficulty of Questions', fontsize=15)

    plt.show()

if __name__ == '__main__':
    # df = pd.read_json('JEOPARDY_QUESTIONS1.json')
    # get_topics(df, 25)
    pass

