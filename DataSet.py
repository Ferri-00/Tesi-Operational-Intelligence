# import libraries
from __future__ import print_function

import pandas as pd
import numpy as np

import time

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

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

def Vectorisation(file_number, max_df, min_df, SVD):
    for v in file_number:
        file_name = f"/home/ATLAS-T3/eferri/File/FrontendFileGroup/storm-frontend-202003{v}-mask-group.csv"
        print('Reading', file_name)
        logs = pd.read_csv(file_name, index_col=0)
        print('creating tokens_per_message')
        tokens_per_message = [x.lower().split() for x in logs.message]
        mean_word_per_message = np.mean([len(x) for x in tokens_per_message])
        print("The average nember o word per message is {}".format(mean_word_per_message))

        c = 0
        for i, dic in enumerate(tokens_per_message):
            if not len(dic):
                print(i, logs.loc[i])
                c += 1

        print("Warning: there are {} blanck messages which will be excluded from the analysis.".format(c))

        # Extract TF-IDF information
        print("Extracting features from the training dataset using a sparse vectorizer with max_df={} and min_df={}".format(max_df, min_df))
        t0 = time()
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,stop_words='english', use_idf=True)
        # vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
        X = vectorizer.fit_transform(logs.message)
        print(vectorizer.stop_words_)

        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % X.shape)
        print()
        
        # Apply LSA for dimensionality reduction to get a lower-dimensional embedding space
        print("Performing dimensionality reduction using LSA to {} dimensions".format(SVD))
        t0 = time()

        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(SVD)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
              int(explained_variance * 100)))

        new_file_name = f'/home/ATLAS-T3/eferri/File/DataSet/data-set-frontend-202003{v}-{SVD}.csv'
        print(f'Saving', new_file_name)
        np.savetxt(new_file_name, X, delimiter=',')        

print("Starting the creation of the data set")
print()
print()

if __name__ == "__main__":
    t0 = time()

    file_number = sys.argv[1]
    max_df = float(sys.argv[2])
    min_df = float(sys.argv[3])
    SVD = int(sys.argv[4])
    print('File number:', file_number)
    print('Max df:', max_df)
    print('Min df:', min_df)
    print('SVD:', SVD)
    Vectorisation([file_number], max_df, min_df, SVD)

    print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")


# ['07','08','09','10','11','12','13']
