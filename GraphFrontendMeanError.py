# import libraries
from __future__ import print_function

import pandas as pd
import numpy as np
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from collections import Counter, defaultdict

import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
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

file_number = ['07', '08']
n_cluster = 28

meanErr = []
meanErrMsg = []
for v in file_number:
    with open(f'/home/ATLAS-T3/eferri/File/DataSet/data-set-frontend-202003{v}.csv') as file_name:
        X = np.loadtxt(file_name, delimiter=",")
    file_name = f"/home/ATLAS-T3/eferri/File/FrontendFileGroup/storm-frontend-202003{v}-mask-group.csv"
    logs = pd.read_csv(file_name, index_col=0, nrows=1e4)

    # run K-Means algorithm: 6 clusters
    km = KMeans(n_clusters=n_cluster, 
                init='k-means++', 
                max_iter=500, 
                n_init=100,
#                 verbose=1
               )

    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("We have {} centroids represented as {}-dimensional points.".format(km.cluster_centers_.shape[0],
                                                                              km.cluster_centers_.shape[1]))    
    # print the numerosity of each cluster
#     print(Counter(km.labels_))

    logs["kmean_labels"] = km.labels_
    label = np.unique(km.labels_)

    error = [0] * len(label)
    error_per_message = [0] * len(label)

    for l in label:
        for msg in logs.message[km.labels_==l]:
            resultE = re.findall('error', msg.lower())
            resultF = re.findall('failure', msg.lower())
            if resultE!=None :
                error_per_message[l] += resultE
                error[l] += 1
            if resultF!=None:
                error_per_message[l] += resultE
                error[l] += 1
        error_per_message[l] /= len(logs.message[km.labels_==l])

    meanErr.append(np.mean(error))
    meanErrMsg.append(np.mean(error_per_message))
    
print('The mean number of message with extracted from files', file_number, 'is', np.mean(meanErr))
print('The mean number of error per message extracted from files', file_number, 'is', np.mean(meanErrMsg))