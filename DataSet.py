# import libraries
from __future__ import print_function

import pandas as pd
import numpy as np

import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

# import logging
from optparse import OptionParser
import sys
from time import time

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from IPython import get_ipython
ipython = get_ipython()

# autoreload extension
# if 'autoreload' not in ipython.extension_manager.loaded:
#     get_ipython().run_line_magic('load_ext', 'autoreload')

# get_ipython().run_line_magic('autoreload', '2')

# Visualizations
import seaborn as sns

def Vectorisation(file_number):
    for v in file_number:
        file_name = f"/home/ATLAS-T3/eferri/File/FrontendFileGroup/storm-frontend-202003{v}-mask-group.csv"
        print('Reading', file_name)
        info = open("/home/ATLAS-T3/eferri/File/DataSet/info.txt", "a")
        info.write('Reading' + file_name + '\n')
	    logs = pd.read_csv(file_name, index_col=0, nrows=1e4)
        print('creating tokens_per_message')
        tokens_per_message = [x.lower().split() for x in logs.message]
        word_set = set()
        
        print('creating word_set')
        info.write('Creating word_set\n')
        for mess in tokens_per_message:
            word_set = word_set.union(set(mess))

        print("We have {} logs messages, for a total of {} unique tokens adopted.".format(
            len(tokens_per_message), len(word_set)))

        word_dict = [dict.fromkeys(word_set, 0) for i in range(len(tokens_per_message))]

        # Compute raw frequencies of each token per each message
        for i in range(len(logs.message)):
            for word in tokens_per_message[i]:
                word_dict[i][word] += 1

        c = 0
        for i, dic in enumerate(tokens_per_message):
            if not len(dic):
                print(i, errors.loc[i])
                c += 1

        print("Warning: there are {} blanck messages which will be excluded from the analysis.".format(c))

        # Extract TF-IDF information
        print("Extracting features from the training dataset using a sparse vectorizer")
        t0 = time()
        vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.02, stop_words='english',
                                     use_idf=True)
        # vectorizer = TfidfVectorizer(stop_words='english',
        #                              use_idf=True)
        X = vectorizer.fit_transform(logs.message)

        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % X.shape)
        print()

        # Apply LSA for dimensionality reduction to get a lower-dimensional embedding space
        print("Performing dimensionality reduction using LSA")
        t0 = time()

        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(25)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
              int(explained_variance * 100)))
        
        print(f'Saving data-set-frontend-202003{v}.csv')
        info.write('Saving data-set-frontend-202003{v}.csv\n')
        np.savetxt(f'/home/ATLAS-T3/eferri/File/DataSet/data-set-frontend-202003{v}.csv', X, delimiter=',')        
        print()
        info.close()

print("Starting the creation of the data set")
t0 = time()

Vectorisation(["07"])
Vectorisation(["08","09","10","11","12", "13"])

print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")

# if __name__ == "__main__":
#     t0 = time()

#     file_number = sys.argv[1]
#     Vectorisation(file_number)

#     print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")

# ['07','08','09','10','11','12','13']
