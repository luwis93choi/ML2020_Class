import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

X, y = make_blobs(n_samples=1000, random_state=1)

agg = AgglomerativeClustering(n_clusters=3)

cluster = agg.fit_predict(X)

cmap = get_cmap('Pastel1')
legend = []
for label in range(agg.n_clusters_):
    
    plt.scatter(X[cluster==label][:, 0], X[cluster==label][:, 1], c=cmap.colors[label], label='Cluster '+ str(label))
    legend.append('Cluster '+ str(label))

plt.legend(legend, loc='best')
plt.title('Agglomertiave Clustering with 3 Clusters')
plt.show()