import numpy as np
import matplotlib.pyplot as plt

from time import time
import sys


def GraphOptimalNCluster(file_number, start=10, stop=50, step=1):
    for v in file_number:
        for type in ['', '-err', '-msg']:
            # with open(f'/home/ATLAS-T3/eferri/File/BestCentroid/frontend-202003{v}{type}-{start}-{stop}-{step}.csv') as file_name:
            with open(f'../File/BestCentroid/frontend-202003{v}{type}-{start}-{stop}-{step}.csv') as file_name:
                X = np.loadtxt(file_name, delimiter=",")

            K=X[0]
            Sum_of_squared_distances=X[1]
            silhouette_avg=X[2]
            calinski_harabasz_avg=X[3]
            davies_bouldin_avg=X[4]

            fig, ax =  plt.subplots(4, 1, figsize=(20, 5))
            ax[0].plot(K,Sum_of_squared_distances,'bx-')
            ax[0].set_xlabel('Values of K') 
            ax[0].set_ylabel('Sum of \n squared \n distances')
            ax[0].grid()
            ax[0].set_title('Elbow Method For Optimal k')

            ax[1].plot(K,silhouette_avg, 'rx-', label='silhouette')
            ax[1].set_xlabel('Values of K') 
            ax[1].set_ylabel('Silhouette \n score') 
            ax[1].grid()
            ax[1].legend()
            # ax[1].set_title('Silhouette analysis For Optimal k')

            ax[2].plot(K,calinski_harabasz_avg, 'gx-', label='calinski harabasz')
            ax[2].set_xlabel('Values of K') 
            ax[2].grid()
            ax[2].legend()

            ax[3].plot(K,davies_bouldin_avg, 'x-', label='davies bouldin')
            ax[3].set_xlabel('Values of K') 
            ax[3].grid()
            ax[3].legend()

            # plt.savefig(f'/home/ATLAS-T3/eferri/File/BestCentroid/frontend-202003{v}{type}-{start}-{stop}-{step}', bbox_inches ="tight")
            plt.savefig(f'../File/BestCentroid/frontend-202003{v}{type}-{start}-{stop}-{step}', bbox_inches ="tight")

if __name__ == "__main__":
    t0 = time()

    file_number = sys.argv[1]
    start = int(sys.argv[2])
    stop = int(sys.argv[3])
    step = int(sys.argv[4])
    print('File number:', file_number)
    print('Arguments:', start, stop, step)
    GraphOptimalNCluster([file_number], start, stop, step)

    print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")

# ['07','08','09','10','11','12','13']
