import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score     # Criterion used to assess the accuracy of clustering (prediction vs cluster label)
from sklearn.metrics.cluster import silhouette_score        # Criterion used to assess the density of clustering (data vs cluster label)

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

X_standardized = StandardScaler().fit_transform(X)

random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

clusterer = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

plt.subplot(1, 4, 1)
cmap = get_cmap('Set1')
legend = []
for label in np.unique(random_clusters):
    
    plt.scatter(X_standardized[random_clusters==label][:, 0], X_standardized[random_clusters==label][:, 1],
                c=cmap.colors[label], label='Cluster '+ str(label))

    legend.append('Cluster '+ str(label))

plt.legend(legend, loc='best')
plt.title('Random Clusters\nARI : {:.6f} / Silhouette Score : {:.2f}'.format(adjusted_rand_score(y, random_clusters), 
                                                                             silhouette_score(X_standardized, random_clusters)))

for i in range(len(clusterer)):
    
    plt.subplot(1, 4, i+2)

    clusters = clusterer[i].fit_predict(X_standardized)

    cmap = get_cmap('Set1')
    legend = []
    for label in np.unique(clusters):
        
        plt.scatter(X_standardized[clusters==label][:, 0], X_standardized[clusters==label][:, 1],
                    c=cmap.colors[label], label='Cluster '+ str(label))

        legend.append('Cluster '+ str(label))

    plt.legend(legend, loc='best')
        
    plt.title('{}\nARI : {:.6f} / Silhouette Score : {:.2f}'.format(clusterer[i].__class__.__name__,
                                                                    adjusted_rand_score(y, clusters), 
                                                                    silhouette_score(X_standardized, clusters)))

plt.show()