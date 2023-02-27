#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Abraham Couperus

# imports
import os
import pandas as pd
import numpy as np
import gensim
import sklearn
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import nltk

# directory to save results
result_dir = "../results/"

# directory of dataset 
data_dir_path = "./tawosi_dataset/"

train_df = pd.DataFrame()
validate_df = pd.DataFrame()
test_df = pd.DataFrame()

def load_data(data_dir_path):
    for file_name in os.listdir(data_dir_path):
        # get training data
        if file_name.endswith("-train.csv"):
            file_path = os.path.join(data_dir_path, file_name)
            # only get title, description_text cols for issue-context
            df = pd.read_csv(file_path, usecols=['title', 'description_text'])
            train_df = pd.concat([train_df, df], ignore_index=True)

        # get validation data
        if file_name.endswith("-valid.csv"):
            file_path = os.path.join(data_dir_path, file_name)
            # only get title, description_text cols for issue-context
            df = pd.read_csv(file_path, usecols=['title', 'description_text'])
            validate_df = pd.concat([validate_df, df], ignore_index=True)
        
        # get testing data
        if file_name.endswith("-test.csv"):
            file_path = os.path.join(data_dir_path, file_name)
            # only get title, description_text cols for issue-context
            df = pd.read_csv(file_path, usecols=['title', 'description_text'])
            test_df = pd.concat([test_df, df], ignore_index=True)


def main():
    load_data(data_dir_path)


if __name__ == '__main__':
    main()