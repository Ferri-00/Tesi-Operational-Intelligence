# import libraries
from __future__ import print_function

import pandas as pd
import numpy as np
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from collections import Counter, defaultdict

import time

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

def Graph(file_number, meanErr, n_cluster):
    for v in file_number:
        with open(f'/home/ATLAS-T3/eferri/File/DataSet/data-set-frontend-202003{v}.csv') as file_name:
            X = np.loadtxt(file_name, delimiter=",")
        file_name = f"/home/ATLAS-T3/eferri/File/FrontendFileErr/storm-frontend-202003{v}-error.csv"
        logs = pd.read_csv(file_name, index_col=0)

        # run K-Means algorithm: ?? clusters
        km = KMeans(n_clusters=n_cluster, 
                    init='k-means++', 
                    max_iter=500, 
                    n_init=100,
                    verbose=1
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
        count = [Counter(km.labels_)[i] for i in label]

        # plot:
        fig, ax = plt.subplots(figsize = (8, int(len(label)/3)))

        for l, c in zip(label, count):
            ax.barh(l, c, linewidth=0.5, edgecolor="white", label=logs.message[km.labels_==l][0])
            ax.text(10, l-0.1, logs.message[logs.kmean_labels==l][0][:80]
    #                 +'\n'+logs.message[logs.kmean_labels==l][0][30:60]
                   )

        ax.set(yticks=label)

        plt.title(f'frontend-202003{v}')
        plt.savefig(f'/home/ATLAS-T3/eferri/File/BestCentroid/frontend-202003{v}', bbox_inches ="tight", facecolor='white')

        # plot:
        fig, ax = plt.subplots(figsize = (8, int(len(label)/3)))

        for l, c in zip(label, count):
            error = sum(logs.error_per_message[logs.kmean_labels == l]) / sum([len(logs.message[logs.kmean_labels == l][i]) 
                                                                              for i in range(len(logs.message[logs.kmean_labels == l]))])
            if error > meanErr:
                color='red'
            else:
                color='green'
            ax.barh(l, c, linewidth=0.5, color=color, edgecolor="white", label=logs.message[logs.kmean_labels==l][0])
            ax.text(10, l-0.2, logs.message[logs.kmean_labels==l][0][:80]
    #                 +'\n'+logs.message[logs.kmean_labels==l][0][30:60]
                   )

        ax.set(yticks=label)
        plt.title(f'frontend-202003{v} '+str(error))
        plt.savefig(f'/home/ATLAS-T3/eferri/File/Graph/frontend-202003{v}', bbox_inches ="tight")
        logs.to_csv(file_name)
        np.savetxt(f'/home/ATLAS-T3/eferri/File/Graph/frontend-202003{v}.csv', km.cluster_centers_, delimiter=',')


if __name__ == "__main__":
    t0 = time()

    file_number = sys.argv[1]
    n_cluster = int(sys.argv[2])
    print('File number:', file_number)
    print('Number of cluster:', n_cluster)
    Graph([file_number], n_cluster=n_cluster)

    print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")

# ['07','08','09','10','11','12','13']
