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
        file_name = f"/home/ATLAS-T3/eferri/File/FrontendFileErr/storm-frontend-202003{v}-msg.csv"
        print('Reading', file_name)
        logs = pd.read_csv(file_name, index_col=0)
        print('creating tokens_per_message')
        tokens_per_message = [x.lower().split() for x in logs.message]
        
        c = 0
        for i, dic in enumerate(tokens_per_message):
            if not len(dic):
                print(i, logs.loc[i])
                c += 1

        print("Warning: there are {} blanck messages which will be excluded from the analysis.".format(c))

        # Extract TF-IDF information
        print("Extracting features from the training dataset using a sparse vectorizer")
        t0 = time()
        vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.05,stop_words='english', use_idf=True)
        # vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
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
        np.savetxt(f'/home/ATLAS-T3/eferri/File/DataSet/data-set-frontend-202003{v}-msg.csv', X, delimiter=',')        
        print()

print("Starting the creation of the data set")
print()
print()

if __name__ == "__main__":
    t0 = time()

    file_number = list(sys.argv[1:])
    Vectorisation(file_number)

    print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")

