#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Abraham Couperus

# imports
import os
import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pickle
import warnings

warnings.filterwarnings('ignore')

import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# extended stop words list taken from ee_clust R code
extended_stop_words = ["a", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaaa", "about", "above", "across", "after", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "an", "and", "another", "any", "anybody", "anyone", "anything", "anywhere", "are",  "aren't", "around", "as", "ask", "asked", "asking", "asks", "at", "away", "b", "back", "be", "became", "because", "become", "becomes", "been", "before", "began", "behind", "being", "beings", "below", "best", "better", "between", "big", "both", "but", "by", "c", "came", "can", "cannot", "can't", "case", "cases", "certain", "certainly", "clear", "clearly", "come", "could", "couldn't", "d", "did", "didn't", "differ", "different", "differently", "do", "does", "doesn't", "doing", "done", "don't", "down", "downed", "downing", "downs", "during", "e", "each", "early", "either", "end", "ended", "ending", "ends", "enough", "even", "evenly", "ever", "every", "everybody", "everyone", "everything", "everywhere", "f", "face", "faces", "fact", "facts", "far", "felt", "few", "find", "finds", "first", "for", "four", "from", "full", "fully", "further", "furthered", "furthering", "furthers", "g", "gave", "general", "generally", "get", "gets", "give", "given", "gives", "go", "going", "good", "goods", "got", "great", "greater", "greatest", "group", "grouped", "grouping", "groups", "h", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's", "high", "higher", "highest", "him", "himself", "his", "how", "however", "how's", "i", "i'd", "if", "i'll", "i'm", "important", "in", "interest", "interested", "interesting", "interests", "into", "is", "isn't", "it", "its", "it's", "itself", "i've", "j", "just", "k", "keep", "keeps", "kind", "knew", "know", "known", "knows", "l", "large", "largely", "last", "later", "latest", "least", "less", "let", "lets", "let's", "like", "likely", "long", "longer", "longest", "m", "made", "make", "making", "man", "many", "may", "me", "member", "members", "men", "might", "more", "most", "mostly", "mr", "mrs", "much", "must", "mustn't", "my", "myself", "n", "necessary", "need", "needed", "needing", "needs", "never", "new", "newer", "newest", "next", "no", "nobody", "non", "noone", "nor", "not", "nothing", "now", "nowhere", "number", "numbers", "o", "of", "off", "often", "old", "older", "oldest", "on", "once", "one", "only", "open", "opened", "opening", "opens", "or", "order", "ordered", "ordering", "orders", "other", "others", "ought", "our", "ours", "ourselves", "out", "over", "own", "p", "part", "parted", "parting", "parts", "per", "perhaps", "place", "places", "point", "pointed", "pointing", "possible", "q", "quite", "r", "rather", "really", "right",  "s", "said", "same", "saw", "say", "says", "see", "seem", "seemed", "seeming", "seems", "sees",  "shall", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",  "since", "small",  "so", "some", "somebody", "someone", "something", "somewhere", "state", "states", "still", "such", "sure", "t", "take", "taken", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "therefore", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "thing", "things", "think", "thinks", "this", "those", "though", "thought", "thoughts", "three", "through", "thus", "to", "today", "together", "too", "took", "toward",  "two", "u", "under", "until", "up", "upon", "us", "use", "used", "uses", "v", "very","via", "w", "want", "wanted", "wanting", "wants", "was", "wasn't", "way", "ways", "we", "we'd", "well", "we'll",  "went", "were", "we're", "weren't", "we've", "what", "what's", "when", "when's", "where", "where's", "whether", "which", "while", "who", "whole", "whom", "who's", "whose", "why", "why's", "will", "with", "within", "without", "won't", "work", "worked", "working", "works", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've", "z"]
extended_stop_words = [word.replace("'", "") for word in extended_stop_words]

# directory to save results
result_dir = "./results3/"

# directory of dataset 
data_dir_path = "./tawosi_dataset/"

all_data = {"train": pd.DataFrame(), "valid": pd.DataFrame(), "test": pd.DataFrame()}
def load_all_data(path, type):
    if type not in ["train", "valid", "test"]:
        print(f"Error loading data with type: {type}. Must be train, valid, or test.")
        return

    dataset = all_data[type]

    for file_name in os.listdir(path):
        project_name = file_name.split("-")[0]
        data_type = re.search(r'(train|valid|test)', file_name)

        file_path = os.path.join(path, file_name)

        if file_name.endswith(f"-{type}.csv"):
            temp = pd.read_csv(file_path, usecols=['issuekey','storypoint','title','description_text'])

            # add issue context
            temp['issue_context'] = temp['title'].str.cat(temp['description_text'], sep=' ')

            # add project name
            temp['project'] = project_name

            # remove title and description_text columns
            temp = temp.drop(columns=['title','description_text'])

            dataset = pd.concat([dataset, temp])

    all_data[type] = dataset
    return dataset


project_data = {"train": pd.DataFrame(), "valid": pd.DataFrame(), "test": pd.DataFrame()}
# Iterates over data files (csv's) and returns the issue context
# for the specified type (train, valid, or test)
def load_project_data(path, type, project, variant = "LHC-SE"):
    if type not in ["train", "valid", "test"]:
        print(f"Error loading data with type: {type}. Must be train, valid, or test.")
        return

    dataset = project_data[type]

    for file_name in os.listdir(path):
        project_name = file_name.split("-")[0]
        data_type = re.search(r'(train|valid|test)', file_name)

        # skip file if not correct type + project
        if data_type.group(0) != type: continue
        if project != project_name: continue

        file_path = os.path.join(path, file_name)

        if file_name == f"{project}-{type}.csv":
            dataset = pd.read_csv(file_path, usecols=['issuekey','storypoint','title','description_text'])

            # add issue context
            dataset['issue_context'] = dataset['title'].str.cat(dataset['description_text'], sep=' ')

            # add project name
            dataset['project'] = project

            # remove title and description_text columns
            dataset = dataset.drop(columns=['title','description_text'])
        
        # add other features (type, component, issue length)
        if variant == "LHC-TC-SE":
            dataset['issue_context_length'] = dataset['issue_context'].apply(len)

            feature_file_name = f"{project_name}-{type}_features.csv"
            file_path = os.path.join(path, feature_file_name)

            if os.path.exists(file_path):
                extra_features = pd.read_csv(file_path)
                extra_features = extra_features.drop(columns=['issuekey'])
                dataset = pd.concat([dataset, extra_features], axis=1)

    project_data[type] = dataset
    return dataset


def get_project_names(path):
    files = os.listdir(path)
    project_names = [file.split("-")[0] for file in files]
    return list(set(project_names))


def remove_urls(text):
    # Remove URLs
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)


def purify(text):
    # Remove all non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9]+', ' ', text)


# Vectorizes and cleans data
# Returns dict of 
# 'data': passed data 
# 'dtm': document term matrix
# 'id2word': id,word pairs for genism LDA model
# 'corpus': bag-of-words representation of issue-context's
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


# Creates and returns LDA model.
# If no number of topics (t) is provided then best t is calculated (time intensive)
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
            eta='auto', # similir to delta
            iterations=500,
            passes=20,
            chunksize=1000,
            eval_every=2,
            gamma_threshold=0.001,
            workers=36
        )

    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken to generate final LDA Model: ", time_taken)

    file_name = f"lda_{t}.model"
    lda_model.save(file_name)
    print(f"Final LDA model saved to {file_name}")

    p = lda_model.log_perplexity(valid['corpus'])
    print("The perplexity of this model is", p)

    return lda_model

# Creates LDA models with various t values
# Returns t value based on model with lowest perplexity
def find_best_t(training, validation):
    # training/validation are sparse matrices

    start_time = time.time()
    # ts = list(range(15, 2001, 500)) # change back to 250
    ts = list(range(5, 16, 5))
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
            passes=20,
            chunksize=1000,
            eval_every=2,
            gamma_threshold=0.001
        )
        lda_model.save(f'lda_{t}.model')
        print(lda_model.print_topics())

        perp = lda_model.log_perplexity(validation['corpus'])
        print(f"Perplexity score at t={t}: {perp}")

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
    plt.savefig("perplexity_graph_new.pdf")

    return int(t['t']), t['perplexity']


## Clustering ##
def get_dtm_lda(training, validation, testing, lda_model):
    train_corpus = vsm(training['issue_context'])['corpus']
    valid_corpus = vsm(validation['issue_context'])['corpus']
    test_corpus = vsm(testing['issue_context'])['corpus']
    
    train_topics = [list(zip(*lda_model.get_document_topics(ic, minimum_probability=0)))[1] for ic in train_corpus]
    valid_topics = [list(zip(*lda_model.get_document_topics(ic, minimum_probability=0)))[1] for ic in valid_corpus]
    test_topics = [list(zip(*lda_model.get_document_topics(ic, minimum_probability=0)))[1] for ic in test_corpus]
    
    return {'train': train_topics, 'valid': valid_topics, 'test': test_topics}


def validate(data, test, dtm_train, dtm_test, eval_method='MdAE'):
    means = data.groupby('labels')['storypoint'].mean()
    medians = data.groupby('labels')['storypoint'].median()

    distance = cdist(dtm_test, dtm_train, metric='cosine')
    closest = np.argmin(distance, axis=1)
    closest_labels = data['labels'].iloc[closest].values

    results = pd.DataFrame({
        'closest': closest,
        'sp': test['storypoint'],
        'closest_sp': data['storypoint'].iloc[closest].values,
        'mean_cluster_sp': means[closest_labels].values,
        'median_cluster_sp': medians[closest_labels].values
    })

    ae_sp_cluster_median = np.abs(results['sp'] - results['median_cluster_sp'])
    mae = np.mean(ae_sp_cluster_median)
    mdae = np.median(ae_sp_cluster_median)

    return {'results': results, 'mae_mdae': (mae, mdae)}


def cluster_h(data, test, valid, dtm, FE="LDA", distance=None, verbose=False, method="ward", ev="sil", project_name=None, lda_model=None):
    if project_name is None:
        project_name = "All_projects"

    dataset_size = dtm['train'].shape[0]
    vocabulary_size = dtm['train'].shape[1]

    if verbose:
        print(f"Evaluation Based-on: {ev}")
        print(f"Corpus Dimensions: {dtm['train'].shape}")

    if distance is None:
        if verbose:
            print("Calculating distance matrix..")
        distance = cdist(dtm['train'], dtm['train'], metric='cosine')
        file_name = f"{result_dir}{project_name}_distance_{FE}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(distance, f)
        print(f"Distance matrix saved to {file_name}")

    dendrogram = linkage(distance, method=method)
    file_name = f"{result_dir}{project_name}_dendrogram_{FE}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(dendrogram, f)
    print(f"Dendrogram saved to {file_name}")

    step = int(dataset_size * 0.1)
    ks = list(range(3, dataset_size - step, step))
    eval_gran = []

    if verbose:
        print("Looping through dendrogram to plot silhouette scores..")

    for i in ks:
        current = cut_tree(dendrogram, n_clusters=i).flatten()
        sil = silhouette_score(distance, current, metric='precomputed')
        data['labels'] = current
        evals = validate(data, valid, dtm['train'], dtm['valid'], eval_method=ev)['mae_mdae']

        eval_gran.append([i, sil, evals[0], evals[1]])

        if verbose:
            print(f"\rDone loop : {i}", end="")

    eval_gran = pd.DataFrame(eval_gran, columns=["granularity", "silhouette", "MAE", "MdAE"])

    file_name = f"{result_dir}{project_name}_gran_{FE}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(eval_gran, f)
    print(f"\nGranularity evaluation table is saved to {file_name}")

    file_name = f"{result_dir}{project_name}_gran_plot_{FE}.pdf"
    plt.plot(eval_gran["granularity"], eval_gran[["silhouette", "MAE", "MdAE"]])
    plt.xlabel("Number of Clusters")
    plt.ylabel("Evaluation Metrics: Silhouette, MAE and MdAE")
    plt.title(f"Cluster Quality for {project_name}")
    plt.legend(["Silhouette", "MAE", "MdAE"])
    plt.savefig(file_name)
    print(f"Plot successfully generated to {file_name}")

    if ev == "sil":
        k = eval_gran.loc[eval_gran["silhouette"].idxmax()]
        print(f"\nBest K is {k['granularity']} Producing {ev} of {k['silhouette']}")
    elif ev == "MAE":
        k = eval_gran.loc[eval_gran["MAE"].idxmin()]
        print(f"\nBest K is {k['granularity']} Producing {ev} of {k['MAE']}")
    elif ev == "MdAE":
        k = eval_gran.loc[eval_gran["MdAE"].idxmin()]
        print(f"\nBest K is {k['granularity']} Producing {ev} of {k['MdAE']}")

    return cut_tree(dendrogram, n_clusters=int(k['granularity'])).flatten()



def main():
    train_data = load_all_data(data_dir_path, type="train")['issue_context']
    valid_data = load_all_data(data_dir_path, type="valid")['issue_context']

    lda(train_data, valid=valid_data, t=2265)

    # Load LDA model
    # lda_model = LdaModel.load("./models/lda_30.model")

    # variant = "LHC-TC-SE"
    # for project_name in get_project_names(data_dir_path):
    #     train_data = load_project_data(data_dir_path, "train", project_name, variant)
    #     valid_data = load_project_data(data_dir_path, "valid", project_name, variant)
    #     test_data = load_project_data(data_dir_path, "test", project_name, variant)

    #     # Fitting LDA model to training, testing and validation data
    #     dtm_lda = get_dtm_lda(train_data, valid_data, test_data, lda_model)

    #     # grab extra features
    #     if variant == "LHC-TC-SE":
    #         train_extra = train_data.drop(columns=['issuekey', 'storypoint', 'issue_context', 'project'])
    #         train_extra = pd.DataFrame(np.reshape(train_extra.values, (train_extra.shape[0], -1)))
    #         assert np.isnan(train_extra.values.astype(np.float64)).sum() == 0, "There are missing values in the data"

    #         valid_extra = valid_data.drop(columns=['issuekey', 'storypoint', 'issue_context', 'project'])
    #         valid_extra = pd.DataFrame(np.reshape(valid_extra.values, (valid_extra.shape[0], -1)))
    #         assert np.isnan(valid_extra.values.astype(np.float64)).sum() == 0, "There are missing values in the data"

    #         test_extra = test_data.drop(columns=['issuekey', 'storypoint', 'issue_context', 'project'])
    #         test_extra = pd.DataFrame(np.reshape(test_extra.values, (test_extra.shape[0], -1)))

    #     # Merge fitted data with extra features if LHC-TC-SE
    #     dtm = {}
    #     if variant == "LHC-TC-SE":
    #         dtm = {}
    #         dtm['train'] = pd.concat([pd.DataFrame(dtm_lda['train']), train_extra], axis=1)
    #         dtm['valid'] = pd.concat([pd.DataFrame(dtm_lda['valid']), valid_extra], axis=1)
    #         dtm['test'] = pd.concat([pd.DataFrame(dtm_lda['test']), test_extra], axis=1)
    #     else: # LHC-SE
    #         dtm = dtm_lda
        
    #     assert dtm['train'].shape[1] == dtm['valid'].shape[1] == dtm['test'].shape[1], "The number of columns in train, valid, and test are not equal"

    #     # perform clustering
    #     train_data['labels'] = cluster_h(train_data, test_data, valid_data, dtm,
    #                 FE = "LDA",
    #                 verbose = True,
    #                 project_name = project_name,
    #                 ev = "MAE",
    #                 lda_model = lda_model)

    #     # find statistics per cluster
    #     results = validate(data=train_data, test=test_data, dtm_train=dtm['train'], dtm_test=dtm['test'])['results']

    #     # Save estimations
    #     results.to_csv(result_dir + project_name + '_results.csv', index=False)

    #     # Print estimation statistics
    #     ae_sp_closest = abs(results['sp'] - results['closest_sp'])
    #     print("\nStory Point - Absolute Error when matching with closest point:\n")
    #     print(ae_sp_closest.describe(include= 'all'))
    #     print("\nMean of Absolute Error: ", ae_sp_closest.mean())
    #     print("Median of Absolute Error: ", ae_sp_closest.median())

    #     ae_sp_cluster_mean = abs(results['sp'] - results['mean_cluster_sp'])
    #     print("\nStory Point - Absolute Error when matching with cluster mean:\n")
    #     print(ae_sp_cluster_mean.describe(include= 'all'))
    #     print("\nMean of Absolute Error: ", ae_sp_cluster_mean.mean())
    #     print("Median of Absolute Error: ", ae_sp_cluster_mean.median())

    #     ae_sp_cluster_median = abs(results['sp'] - results['median_cluster_sp'])
    #     print("\nStory Point - Absolute Error when matching with cluster median:\n")
    #     print(ae_sp_cluster_median.describe(include= 'all'))
    #     print("\nMean of Absolute Error: ", ae_sp_cluster_median.mean())
    #     print("Median of Absolute Error: ", ae_sp_cluster_median.median())

    #     print("\n########################################################################\n")
    

if __name__ == '__main__':
    main()