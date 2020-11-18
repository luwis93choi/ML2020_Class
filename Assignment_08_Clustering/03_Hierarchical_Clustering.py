import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from scipy.cluster.hierarchy import dendrogram, ward

#############################################
### Dendrogram of hierarchical clustering ###
#############################################

### Prepare random dataset with blob distribution ##################################################################################################
X, y = make_blobs(random_state=0, n_samples=12)

### Clustering & Plotting hierarchy of cluster merging #############################################################################################
linkage_array = ward(X)     # Use ward in scipy library for hierarchical clustering

dendrogram(linkage_array)   # Plot denrogram of hierarchical clustering in order to show the hierarchical merging of clusters

### Write the values for axes ######################################################################################################################
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' 2 clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' 3 clusters', va='center', fontdict={'size': 15})
plt.xlabel("sample number")
plt.ylabel("cluster distance")

plt.show()