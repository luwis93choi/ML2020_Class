import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

##################################################
### Agglomerative clustering on random dataset ###
##################################################

### Prepare random dataset with blob distribution ##################################################################################################
X, y = make_blobs(n_samples=1000, random_state=1)

### Agglomertiave Clustering Preparation ###########################################################################################################
agg = AgglomerativeClustering(n_clusters=3)     # Agglomerative clusterer with 3 clusters

cluster = agg.fit_predict(X)    # Fit agglomerative clusterer with current random dataset

### Plot clustering results of agglomerative clustering on given random dataset #####################################################################
cmap = get_cmap('Pastel1')      # Prepare color map / Each cluster uses an distinctive color
legend = []
for label in range(agg.n_clusters_):
    
    # Plot only the points that correspond to certain cluster label using X[cluster==label]
    # Assign the color to the points in the dataset according to their labels
    plt.scatter(X[cluster==label][:, 0], X[cluster==label][:, 1], c=cmap.colors[label], label='Cluster '+ str(label))
    legend.append('Cluster '+ str(label))

plt.legend(legend, loc='best')
plt.title('Agglomertiave Clustering with 3 Clusters')
plt.show()