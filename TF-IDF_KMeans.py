# import libraries
from __future__ import print_function

import pandas as pd
import numpy as np

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

def centroid(file_number):
    for v in file_number:
        file_name = f"../File/FrontendFileGroup/storm-frontend-202003{v}-mask-group.txt"
        # file_name = f"../File/FrontendFileGroup/storm-frontend-202003{v}-group.txt"

        logs = pd.read_csv(file_name, index_col= 0)

        tokens_per_message = [x.lower().split() for x in logs.message]

        word_set = set()
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

        def compute_tf(word_dict, l): # l is the message from logs
            tf = {}
            sum_nk = len(l)
            for word, count in word_dict.items():
                try:
                    tf[word] = count/sum_nk
                except ZeroDivisionError:
                    tf[word] = 0
            return tf

        tf = [compute_tf(word_dict[i], tokens_per_message[i])
              for i in range(len(tokens_per_message))] #if sum(word_dict[i].values())]

        def compute_idf(strings_list):
            n = len(strings_list)
            idf = dict.fromkeys(strings_list[0].keys(), 0)
            for l in strings_list:
                for word, count in l.items():
                    if count > 0:
                        idf[word] += 1

            for word, v in idf.items():
                idf[word] = np.log(n / float(v))
            return idf

        idf = compute_idf(word_dict)

        def compute_tf_idf(tf, idf):
            tf_idf = dict.fromkeys(tf.keys(), 0)
            for word, v in tf.items():
                tf_idf[word] = v * idf[word]
            return tf_idf

        tf_idf =  [compute_tf_idf(tf[i], idf) for i in range(len(tf))]

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

        print()

        labels = np.arrange(0, 29)
        counts = [[0] * 7]*30
        cluster = [0] *7
        
        c=0
        for n_clusters in range(10, 30, 3):
            cluster[c] = n_clusters
            # run K-Means algorithm: 6 clusters
            km = KMeans(n_clusters=n_clusters, 
                        init='k-means++', 
                        max_iter=500, 
                        n_init=100,
                        verbose=1)

            print("Clustering sparse data with %s" % km)
            t0 = time()
            km.fit(X)
            print("done in %0.3fs" % (time() - t0))
            print()
        
            print("We have {} centroids represented as {}-dimensional points.".format(km.cluster_centers_.shape[0],
                                                                                      km.cluster_centers_.shape[1]))
            # print the numerosity of each cluster
            print(Counter(km.labels_))

            logs["kmean_labels"] = km.labels_


            label = np.unique(km.labels_)
            count = [Counter(km.labels_)[i] for i in label]

    #         for l in label:
    #             print(logs.message[logs.kmean_labels==l][0:5], '\n')
    #             with open('m')
            plt.style.use('_mpl-gallery')

            # plot:
            fig, ax = plt.subplots(figsize = (8, 5))

            for l, c in zip(label, count):
                ax.barh(l, c, linewidth=0.5, edgecolor="white", label=logs.message[logs.kmean_labels==l][0])
                ax.text(c, l-0.35, logs.message[logs.kmean_labels==l][0][:30]+'\n'+logs.message[logs.kmean_labels==l][0][30:60])

            ax.set(yticks=label)

            plt.title(f'frontend-202003{v}-{n_clusters}centroids')
            plt.savefig(f'frontend-202003{v}-{n_clusters}centroids', bbox_inches ="tight")
            
            for i in range(len(label)):
                counts[i][c] = count[i]
            c+=1          
        
        # plot:
        fig, ax = plt.subplots(figsize = (8, 5))
        color = list(mcolors.CSS4_COLORS)

        col=10
        for l, c in zip(labels, counts):
            ax.plot(cluster, c, '.', color=color[col], label=l)
            col+=2

        ax.set(xticks=label)

        ax.legend()
        plt.title(f'frontend-202003{v}-{n_clusters}centroids  '+str(error))
        plt.savefig(f'frontend-202003{v}_best_centroids', bbox_inches ="tight")

print("Starting best centroid number analisys")

t0 = time()

centroid(['07'])
# centroid(['08', '09', '10', '11', '12', '13'])

print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")