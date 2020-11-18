import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

import mglearn

##############################
### DBSCAN example display ###
##############################

mglearn.plots.plot_dbscan()  # DBSCAN results according to the variations in min_samples and eps (distance threshold for cluster merging)
plt.show()

####################################################################
### DBSCAN comparison under different parameters in moon dataset ###
####################################################################

### Prepare moon dataset ##########################################################################################################################
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)  # Moon dataset with 1000 samples

### DBSCAN clustering of moon dataset using different parameters ###################################################################################

# DBSCAN 1 (eps : 0.05 / min_sample : 5 / distance metric : euclidean) #############################################################################
dbscan1 = DBSCAN(eps=0.05, min_samples=5, metric='euclidean')   
db_cluster1 = dbscan1.fit_predict(X)    # Fit and cluster the dataset using DBSCAN 1

print('DBSCAN core sample num : {}'.format(len(dbscan1.core_sample_indices_)))
print('DBSCAN Labels : {}'.format(np.unique(dbscan1.labels_)))

# Plot clustering results of DBSCAN 1
plt.subplot(121)
cmap = get_cmap('Accent')   # Prepare color map / Each cluster uses an distinctive color
legend = []
for label in np.unique(dbscan1.labels_):
    
    # Plot only the points that correspond to certain cluster label using X[db_cluster1==label]
    # Assign the color to the points in the dataset according to their labels
    plt.scatter(X[db_cluster1==label][:, 0], X[db_cluster1==label][:, 1], c=cmap.colors[label], label='Cluster '+ str(label))
    legend.append('Cluster '+ str(label))

plt.legend(legend, loc='best')
plt.title('DBSCAN (eps : {} / min_sample : {})'.format(dbscan1.eps, dbscan1.min_samples))

# DBSCAN 2 (eps : 0.2 / min_sample : 5 / distance metric : euclidean) #############################################################################
dbscan2 = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
db_cluster2 = dbscan2.fit_predict(X)    # Fit and cluster the dataset using DBSCAN 1

print('DBSCAN core sample num : {}'.format(len(dbscan2.core_sample_indices_)))
print('DBSCAN Labels : {}'.format(np.unique(dbscan2.labels_)))

# Plot clustering results of DBSCAN 2
plt.subplot(122)
cmap = get_cmap('Accent')   # Prepare color map / Each cluster uses an distinctive color
legend = []
for label in np.unique(dbscan2.labels_):
    
    # Plot only the points that correspond to certain cluster label using X[db_cluster2==label]
    # Assign the color to the points in the dataset according to their labels
    plt.scatter(X[db_cluster2==label][:, 0], X[db_cluster2==label][:, 1], c=cmap.colors[label], label='Cluster '+ str(label))
    legend.append('Cluster '+ str(label))

plt.legend(legend, loc='best')
plt.title('DBSCAN (eps : {} / min_sample : {})'.format(dbscan2.eps, dbscan2.min_samples))

plt.show()