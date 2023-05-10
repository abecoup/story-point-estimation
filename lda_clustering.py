#!/usr/bin/env python
# -*- coding: utf-8 -*-

# AUTHOR: Abraham Couperus

# imports
import os
import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
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
import csv

warnings.filterwarnings('ignore')

import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# extended stop words list taken from ee_clust R code
extended_stop_words = ["a", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaaa", "about", "above", "across", "after", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "an", "and", "another", "any", "anybody", "anyone", "anything", "anywhere", "are",  "aren't", "around", "as", "ask", "asked", "asking", "asks", "at", "away", "b", "back", "be", "became", "because", "become", "becomes", "been", "before", "began", "behind", "being", "beings", "below", "best", "better", "between", "big", "both", "but", "by", "c", "came", "can", "cannot", "can't", "case", "cases", "certain", "certainly", "clear", "clearly", "come", "could", "couldn't", "d", "did", "didn't", "differ", "different", "differently", "do", "does", "doesn't", "doing", "done", "don't", "down", "downed", "downing", "downs", "during", "e", "each", "early", "either", "end", "ended", "ending", "ends", "enough", "even", "evenly", "ever", "every", "everybody", "everyone", "everything", "everywhere", "f", "face", "faces", "fact", "facts", "far", "felt", "few", "find", "finds", "first", "for", "four", "from", "full", "fully", "further", "furthered", "furthering", "furthers", "g", "gave", "general", "generally", "get", "gets", "give", "given", "gives", "go", "going", "good", "goods", "got", "great", "greater", "greatest", "group", "grouped", "grouping", "groups", "h", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's", "high", "higher", "highest", "him", "himself", "his", "how", "however", "how's", "i", "i'd", "if", "i'll", "i'm", "important", "in", "interest", "interested", "interesting", "interests", "into", "is", "isn't", "it", "its", "it's", "itself", "i've", "j", "just", "k", "keep", "keeps", "kind", "knew", "know", "known", "knows", "l", "large", "largely", "last", "later", "latest", "least", "less", "let", "lets", "let's", "like", "likely", "long", "longer", "longest", "m", "made", "make", "making", "man", "many", "may", "me", "member", "members", "men", "might", "more", "most", "mostly", "mr", "mrs", "much", "must", "mustn't", "my", "myself", "n", "necessary", "need", "needed", "needing", "needs", "never", "new", "newer", "newest", "next", "no", "nobody", "non", "noone", "nor", "not", "nothing", "now", "nowhere", "number", "numbers", "o", "of", "off", "often", "old", "older", "oldest", "on", "once", "one", "only", "open", "opened", "opening", "opens", "or", "order", "ordered", "ordering", "orders", "other", "others", "ought", "our", "ours", "ourselves", "out", "over", "own", "p", "part", "parted", "parting", "parts", "per", "perhaps", "place", "places", "point", "pointed", "pointing", "possible", "q", "quite", "r", "rather", "really", "right",  "s", "said", "same", "saw", "say", "says", "see", "seem", "seemed", "seeming", "seems", "sees",  "shall", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",  "since", "small",  "so", "some", "somebody", "someone", "something", "somewhere", "state", "states", "still", "such", "sure", "t", "take", "taken", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "therefore", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "thing", "things", "think", "thinks", "this", "those", "though", "thought", "thoughts", "three", "through", "thus", "to", "today", "together", "too", "took", "toward",  "two", "u", "under", "until", "up", "upon", "us", "use", "used", "uses", "v", "very","via", "w", "want", "wanted", "wanting", "wants", "was", "wasn't", "way", "ways", "we", "we'd", "well", "we'll",  "went", "were", "we're", "weren't", "we've", "what", "what's", "when", "when's", "where", "where's", "whether", "which", "while", "who", "whole", "whom", "who's", "whose", "why", "why's", "will", "with", "within", "without", "won't", "work", "worked", "working", "works", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've", "z"]
extended_stop_words = [word.replace("'", "") for word in extended_stop_words]


# Given path to data, parses list of project names
def get_project_names(path):
    files = os.listdir(path)
    project_names = [file.split("-")[0] for file in files]
    return list(set(project_names))


# Used to save MAE and MdAE metrics
def save_project_metrics(project_name, mae, mdae, variant):
    file_name = f"project_metrics_{variant}.csv"

    # Check if the file exists, if not, create it and write the header
    try:
        with open(file_name, "x") as f:
            writer = csv.writer(f)
            writer.writerow(["project_name", "mae", "mdae"])
    except FileExistsError:
        pass

    # Append the project metrics to the file
    with open(file_name, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([project_name, mae, mdae])


# Remove all non-alphanumeric characters
def purify(text):
    return re.sub(r'[^a-zA-Z0-9]+', ' ', text)


def remove_urls(text):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)


# Gets all data based on type (train, valid, test), no project seperation
def load_all_data(path, type):
    if type not in ["train", "valid", "test"]:
        print(f"Error loading data with type: {type}. Must be train, valid, or test.")
        return

    dataset = pd.DataFrame()

    for file_name in os.listdir(path):
        project_name = file_name.split("-")[0]
        file_path = os.path.join(path, file_name)

        if file_name.endswith(f"-{type}.csv"):
            temp = pd.read_csv(file_path, usecols=['issuekey','storypoint','title','description_text'])

            # add issue context (title + description)
            temp['issue_context'] = temp['title'].str.cat(temp['description_text'], sep=' ')

            # remove title and description cols
            temp = temp.drop(columns=['title','description_text'])

            # add project name
            temp['project'] = project_name

            dataset = pd.concat([dataset, temp])

    return dataset


# Gets data based on project name AND type (train, valid, test)
# If variant is LHC-TC-SE, adds extra features
def load_project_data(path, type, project, variant = "LHC-SE"):
    if type not in ["train", "valid", "test"]:
        print(f"Error loading data with type: {type}. Must be train, valid, or test.")
        return

    dataset = pd.DataFrame()

    for file_name in os.listdir(path):
        project_name = file_name.split("-")[0]
        data_type = re.search(r'(train|valid|test)', file_name)

        # skip file if not correct type + project
        if data_type.group(0) != type: continue
        if project != project_name: continue

        file_path = os.path.join(path, file_name)

        if file_name == f"{project}-{type}.csv":
            dataset = pd.read_csv(file_path, usecols=['issuekey','storypoint','title','description_text'])

            # add issue context (title + description)
            dataset['issue_context'] = dataset['title'].str.cat(dataset['description_text'], sep=' ')

            # remove title and description cols
            dataset = dataset.drop(columns=['title','description_text'])

            # add project name
            dataset['project'] = project
        
        # add other features (type, component, issue length)
        if variant == "LHC-TC-SE":
            dataset['issue_context_length'] = dataset['issue_context'].apply(len)

            feature_file_name = f"{project_name}-{type}_features.csv"
            file_path = os.path.join(path, feature_file_name)

            if os.path.exists(file_path):
                extra_features = pd.read_csv(file_path)
                extra_features = extra_features.drop(columns=['issuekey'])
                dataset = pd.concat([dataset, extra_features], axis=1)

    return dataset


# Performs pre-processing, tokenization, and vectorization of data
def create_vector_space_model(data):

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
    
    return {'data': data, # pre-processed text
            'dtm': doc_term_matrix, # document-term matrix
            'id2word': id2word, # Gensim dictionary
            'corpus': corpus} # Gensim corpus


# Creates and returns Genism LDA model.
# If no number of topics (t) is provided then best t is calculated (time intensive)
def create_lda_model(train, valid, t=None):
    print("Generating LDA Model..\n")

    train = create_vector_space_model(train)
    valid = create_vector_space_model(valid)

    # time/compute intensive
    if t is None:
        t_res = find_best_t(train, valid)
        print("Best t: ", t_res[0], " produced perplexity: ", t_res[1])
        t = t_res[0]
    
    start_time = time.time()

    # Build LDA Model
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
            gamma_threshold=0.001
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


# Creates LDA models with various t values (8 models total)
# Returns t value based on model with lowest perplexity
def find_best_t(training, validation):

    start_time = time.time()
    ts = list(range(15, 2016, 250)) # t value range
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
            eta='auto', # similir to delta
            iterations=500,
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


# Extracts the topic distributions for the training, validation, and testing datasets using the provided LDA model.
# Returns a dictionary containing the topic distributions for each dataset.
def extract_topic_distributions(training, validation, testing, lda_model):
    train_corpus = create_vector_space_model(training['issue_context'])['corpus']
    valid_corpus = create_vector_space_model(validation['issue_context'])['corpus']
    test_corpus = create_vector_space_model(testing['issue_context'])['corpus']
    
    train_topics = [list(zip(*lda_model.get_document_topics(ic, minimum_probability=0)))[1] for ic in train_corpus]
    valid_topics = [list(zip(*lda_model.get_document_topics(ic, minimum_probability=0)))[1] for ic in valid_corpus]
    test_topics = [list(zip(*lda_model.get_document_topics(ic, minimum_probability=0)))[1] for ic in test_corpus]
    
    return {'train': train_topics, 'valid': valid_topics, 'test': test_topics}


# Calculates the mean and median absolute errors between the actual storypoints of test documents and the median
# storypoints of the closest clusters. Returns a dictionary containing the evaluation results and the mean and
# median absolute errors.
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


# Performs hierarchical clustering on the input data, iterates through different granularities (number of clusters),
# and calculates evaluation metrics (silhouette score, MAE, and MdAE) for each granularity. Saves the evaluation
# metrics in a table and plots them against the number of clusters. Returns the best clustering labels based on the
# chosen evaluation metric (silhouette, MAE, or MdAE).
def perform_clustering(data, test, valid, dtm, FE="LDA", distance=None, verbose=False, method="ward", ev="sil", project_name=None, lda_model=None, result_dir="./"):
    if project_name is None:
        project_name = "All_projects"

    dataset_size = dtm['train'].shape[0]
    vocabulary_size = dtm['train'].shape[1]

    if verbose:
        print(f"Evaluation Based-on: {ev}")
        print(f"Corpus Dimensions: {dtm['train'].shape}")

    # calculate cosine distance
    if distance is None:
        if verbose:
            print("Calculating distance matrix..")
        distance = cdist(dtm['train'], dtm['train'], metric='cosine')
        file_name = f"{result_dir}{project_name}_distance_{FE}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(distance, f)
        print(f"Distance matrix saved to {file_name}")

    # Perform hierarchical clustering
    dendrogram = linkage(distance, method=method)
    
    # Save dendrogram plot
    plt.figure(figsize=(12, 6))
    try:
        scipy_dendrogram(dendrogram)
        plt.title(f"Dendrogram for {project_name}")
        plt.xlabel("Issue-Context Distance Vectors")
        plt.ylabel("Distance Measure")

        # Save the dendrogram plot to a file
        dendrogram_file_name = f"{result_dir}{project_name}_dendrogram_plot_{FE}.png"
        plt.savefig(dendrogram_file_name)
        plt.close()
        print(f"Dendrogram plot saved to {dendrogram_file_name}")
    except RecursionError:
        print("Warning: Maximum recursion depth reached. Dendrogram plot skipped.")

    # Determine optimal granularity (number of clusters) using evaluation metrics
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

    # Save evaluation metrics table in pickle file
    file_name = f"{result_dir}{project_name}_gran_{FE}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(eval_gran, f)
    print(f"Evaluation metrics table saved to {file_name}")

    # Save evaluation metrics plot as a PDF
    file_name = f"{result_dir}{project_name}_gran_plot_{FE}.pdf"
    plt.plot(eval_gran["granularity"], eval_gran[["silhouette", "MAE", "MdAE"]])
    plt.xlabel("Number of Clusters")
    plt.ylabel("Evaluation Metrics: Silhouette, MAE and MdAE")
    plt.title(f"Cluster Quality for {project_name}")
    plt.legend(["Silhouette", "MAE", "MdAE"])
    plt.savefig(file_name)
    plt.close()
    print(f"Plot successfully generated to {file_name}")

    # Determine best number of clusters (k) based on evaluation metric
    if ev == "sil":
        k = eval_gran.loc[eval_gran["silhouette"].idxmax()]
        print(f"\nBest K is {k['granularity']} Producing {ev} of {k['silhouette']}")
    elif ev == "MAE":
        k = eval_gran.loc[eval_gran["MAE"].idxmin()]
        print(f"\nBest K is {k['granularity']} Producing {ev} of {k['MAE']}")
    elif ev == "MdAE":
        k = eval_gran.loc[eval_gran["MdAE"].idxmin()]
        print(f"\nBest K is {k['granularity']} Producing {ev} of {k['MdAE']}")

    # Cut dendrogram at k, returns clustering labels
    return cut_tree(dendrogram, n_clusters=int(k['granularity'])).flatten()