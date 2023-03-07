#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Abraham Couperus

# imports
import os
import pandas as pd
import numpy as np
import time
import re
import gensim
import sklearn
import nltk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.ldamodel import LdaModel
from gensim.matutils import corpus2csc
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus

# extended stop words list taken from ee_clust R code
extended_stop_words = ["a", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaaa", "about", "above", "across", "after", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "an", "and", "another", "any", "anybody", "anyone", "anything", "anywhere", "are",  "aren't", "around", "as", "ask", "asked", "asking", "asks", "at", "away", "b", "back", "be", "became", "because", "become", "becomes", "been", "before", "began", "behind", "being", "beings", "below", "best", "better", "between", "big", "both", "but", "by", "c", "came", "can", "cannot", "can't", "case", "cases", "certain", "certainly", "clear", "clearly", "come", "could", "couldn't", "d", "did", "didn't", "differ", "different", "differently", "do", "does", "doesn't", "doing", "done", "don't", "down", "downed", "downing", "downs", "during", "e", "each", "early", "either", "end", "ended", "ending", "ends", "enough", "even", "evenly", "ever", "every", "everybody", "everyone", "everything", "everywhere", "f", "face", "faces", "fact", "facts", "far", "felt", "few", "find", "finds", "first", "for", "four", "from", "full", "fully", "further", "furthered", "furthering", "furthers", "g", "gave", "general", "generally", "get", "gets", "give", "given", "gives", "go", "going", "good", "goods", "got", "great", "greater", "greatest", "group", "grouped", "grouping", "groups", "h", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's", "high", "higher", "highest", "him", "himself", "his", "how", "however", "how's", "i", "i'd", "if", "i'll", "i'm", "important", "in", "interest", "interested", "interesting", "interests", "into", "is", "isn't", "it", "its", "it's", "itself", "i've", "j", "just", "k", "keep", "keeps", "kind", "knew", "know", "known", "knows", "l", "large", "largely", "last", "later", "latest", "least", "less", "let", "lets", "let's", "like", "likely", "long", "longer", "longest", "m", "made", "make", "making", "man", "many", "may", "me", "member", "members", "men", "might", "more", "most", "mostly", "mr", "mrs", "much", "must", "mustn't", "my", "myself", "n", "necessary", "need", "needed", "needing", "needs", "never", "new", "newer", "newest", "next", "no", "nobody", "non", "noone", "nor", "not", "nothing", "now", "nowhere", "number", "numbers", "o", "of", "off", "often", "old", "older", "oldest", "on", "once", "one", "only", "open", "opened", "opening", "opens", "or", "order", "ordered", "ordering", "orders", "other", "others", "ought", "our", "ours", "ourselves", "out", "over", "own", "p", "part", "parted", "parting", "parts", "per", "perhaps", "place", "places", "point", "pointed", "pointing", "possible", "q", "quite", "r", "rather", "really", "right",  "s", "said", "same", "saw", "say", "says", "see", "seem", "seemed", "seeming", "seems", "sees",  "shall", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",  "since", "small",  "so", "some", "somebody", "someone", "something", "somewhere", "state", "states", "still", "such", "sure", "t", "take", "taken", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "therefore", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "thing", "things", "think", "thinks", "this", "those", "though", "thought", "thoughts", "three", "through", "thus", "to", "today", "together", "too", "took", "toward",  "two", "u", "under", "until", "up", "upon", "us", "use", "used", "uses", "v", "very","via", "w", "want", "wanted", "wanting", "wants", "was", "wasn't", "way", "ways", "we", "we'd", "well", "we'll",  "went", "were", "we're", "weren't", "we've", "what", "what's", "when", "when's", "where", "where's", "whether", "which", "while", "who", "whole", "whom", "who's", "whose", "why", "why's", "will", "with", "within", "without", "won't", "work", "worked", "working", "works", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've", "z"]
extended_stop_words = [word.replace("'", "") for word in extended_stop_words]

# directory to save results
result_dir = "../results/"

# directory of dataset 
data_dir_path = "./tawosi_dataset/"

data_frames = {"train": pd.DataFrame(), "valid": pd.DataFrame(), "test": pd.DataFrame()}

# Iterates over data files (csv's) and returns the issue context
# for the specified data_type (train, valid, or test)
def load_data(data_dir_path, data_type):
    if data_type not in ["train", "valid", "test"]:
        print(f"Error loading data with type: {data_type}. Must be train, valid, or test.")
        return

    file_suffix = f"-{data_type}.csv"
    data_frame = data_frames[data_type]
    for file_name in os.listdir(data_dir_path):
        if file_name.endswith(file_suffix):
            file_path = os.path.join(data_dir_path, file_name)
            df = pd.read_csv(file_path, usecols=['title', 'description_text'])
            df['issue_context'] = df['title'].str.cat(df['description_text'], sep=' ')
            data_frame = pd.concat([data_frame, df], ignore_index=True)
    
    data_frames[data_type] = data_frame
    return data_frame[['issue_context']]


def remove_urls(text):
    # Remove URLs
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)


def purify(text):
    # Remove all non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9]+', ' ', text)


# Function that builds the vector space
def vsm(data, weighting='tf'):

    # Remove URLs
    data = [remove_urls(issue_context) for issue_context in data]

    # Remove non-alphanumeric characters
    data = [purify(issue_context) for issue_context in data]

    # Remove empty issue-contexts
    data = [ic for ic in data if len(ic) > 0]

    # The rest of the pre-processing happens in CountVectorizer, thus they are passed in using a CountVectorizer object
    # By default CountVectorizer makes lower case, removes punctuation, and only considers words with at least two characters
    vectorizer = CountVectorizer(stop_words = stopwords.words('english') + extended_stop_words,
                                    min_df=1, # exclude any terms that do not appear in any document in the corpus
                                    )
    doc_term_matrix = vectorizer.fit_transform(data)

    # Convert the document-term matrix to Gensim's corpus format
    corpus = Sparse2Corpus(doc_term_matrix, documents_columns=False)

    # Create a dictionary mapping words to their integer ids
    id2word = Dictionary.from_corpus(corpus, id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

    term_freqs = doc_term_matrix.sum(axis=0)
    assert not np.any(term_freqs == 0), "Some terms have zero frequency."
    doc_lens = doc_term_matrix.sum(axis=1)
    assert not np.any(doc_lens == 0), "Some documents have zero length."
    
    return {'data': data,
            'dtm': doc_term_matrix,
            'id2word': id2word,
            'corpus': corpus}


# Function that builds a vector space out of LDA topics.
# t is number of topics (if known), leave null to calculate the t that produces 
# the least perplexity.
# if t is null, need to send validation data as well.
def lda(train, valid=None, t=None):
    print("Generating LDA Model..\n")

    train = vsm(train)
    valid = vsm(valid)

    if t is None:
        t_res = find_best_t(train, valid)
        print("Best t: ", t_res[0], " produced perplexity: ", t_res[1])
        t = t_res[0]
    
    start_time = time.time()

    lda_model = LdaModel(
            corpus=Sparse2Corpus(train['dtm']),
            id2word=train['id2word'],
            num_topics=t,
            alpha=1/t,
            eta=0.1, # similir to delta
            iterations=500,
            passes=50,
            eval_every=100
        )

    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken to generate final LDA Model: ", time_taken)

    file_name = f"lda_{t}.rda"
    lda_model.save(file_name)
    print(f"Final LDA model saved to {file_name}")

    p = lda_model.log_perplexity(valid['dtm'])
    print("The perplexity of this model is", np.exp2(-p))

    return lda_model


def find_best_t(training, validation):
    # training/validation are sparse matrices

    start_time = time.time()
    ts = list(range(15, 2001, 500)) # change back to 250
    models = []
    corpus = training['corpus']
    id2word = training['id2word']

    for t in ts:
        print("Generating LDA model for t=" + str(t))
        lda_model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=t,
            alpha=1/t,
            eta=0.1, # similir to delta
            iterations=300,
            passes=50
        )
        models.append(lda_model)

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to generate LDA models: {time_taken:.2f} seconds")

    perps = [model.log_perplexity(validation['corpus']) for model in models]
    t_perps = list(zip(ts, perps))
    t_perps_df = pd.DataFrame(t_perps, columns=['t', 'perplexity'])
    t = t_perps_df.loc[t_perps_df['perplexity'].idxmin()]

    plt.plot(ts, perps)
    plt.xlabel("Number of topics")
    plt.ylabel("Perplexity")
    plt.savefig("perplexity_graph.pdf")

    return int(t['t']), np.exp(-t['perplexity'])



def main():
    train_data = load_data(data_dir_path, "train")
    valid_data = load_data(data_dir_path, "valid")
    lda(train_data['issue_context'], valid_data['issue_context'])
    


if __name__ == '__main__':
    main()