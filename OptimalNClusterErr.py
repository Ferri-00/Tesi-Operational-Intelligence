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
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


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

def BestCentroid(file_number, SVD, start=10, stop=50, step=1):
    silhouette = []
    centroid=[]
    for v in file_number:
        with open(f'/home/ATLAS-T3/eferri/File/DataSet/data-set-frontend-202003{v}-{SVD}-err.csv') as file_name:
            X = np.loadtxt(file_name, delimiter=",")

        Sum_of_squared_distances = []
        silhouette_avg = []
        calinski_harabasz_avg = []
        davies_bouldin_avg = []

        K = range(start, stop, step)    

        for n_clusters in K:
            # run K-Means algorithm: 6 clusters
            km = KMeans(n_clusters=n_clusters, 
                        init='k-means++', 
                        max_iter=500, 
                        n_init=100,
                        verbose=1
                       )

            print("Clustering sparse data with %s" % km)
            t0 = time()
            km.fit(X)
            print("done in %0.3fs" % (time() - t0))

            print("We have {} centroids represented as {}-dimensional points.".format(km.cluster_centers_.shape[0],
                                                                                      km.cluster_centers_.shape[1]))
            print()

            Sum_of_squared_distances.append(km.inertia_)
            cluster_labels = km.labels_
            # silhouette score
            silhouette_avg.append(silhouette_score(X, cluster_labels, metric='euclidean'))
            calinski_harabasz_avg.append(calinski_harabasz_score(X, cluster_labels))
            davies_bouldin_avg.append(davies_bouldin_score(X, cluster_labels))

            label = np.unique(km.labels_)
            count = [Counter(km.labels_)[i] for i in label]

        np.savetxt(f'/home/ATLAS-T3/eferri/File/BestCentroid/frontend-202003{v}-{SVD}-err-{start}-{stop}-{step}.csv',
                   (K, Sum_of_squared_distances, silhouette_avg,calinski_harabasz_avg,davies_bouldin_avg),
                   delimiter=',')

        fig, ax =  plt.subplots(4, 1, figsize=(20, 6))
        ax[0].plot(K,Sum_of_squared_distances,'bx-', label='Elbow Method')
        ax[0].set_ylabel('Sum of \n squared \n distances')
        ax[0].set_title(f'Best centroid n. for frontend-err {v}')
        
        ax[1].plot(K,silhouette_avg, 'rx-', label='silhouette')
        ax[1].set_ylabel('Silhouette \n score') 
#         ax[1].set_title('Silhouette analysis For Optimal k')
        
        ax[2].plot(K,calinski_harabasz_avg, 'gx-', label='calinski harabasz')

        ax[3].plot(K,davies_bouldin_avg, 'x-', label='davies bouldin')
        
        for i in range(4):
            ax[i].set_xlabel('Values of K')
            ax[i].grid()
            ax[i].legend()
            ax[i].set_xticks(np.arange(2, 20, step=1))

        plt.savefig(f'/home/ATLAS-T3/eferri/File/BestCentroid/frontend-202003{v}-{SVD}-err-{start}-{stop}-{step}', bbox_inches ="tight")

if __name__ == "__main__":
    t0 = time()

    file_number = sys.argv[1]
    start = int(sys.argv[2])
    stop = int(sys.argv[3])
    step = int(sys.argv[4])
    SVD = int(sys.argv[5])
    print('File number:', file_number)
    print('Arguments:', start, stop, step, SVD)
    BestCentroid([file_number], SVD, start, stop, step)

    print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")
