import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

import time

####################################################################################################################################################
### Prepare MNIST dataset ##########################################################################################################################
####################################################################################################################################################
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

X = mnist['data']       # Load data from MNIST dataset
y = mnist['target']     # Load target values from MNIST dataset

X_train, X_test, y_train, y_test = train_test_split(X, y)   # Split the dataset between training and test

####################################################################################################################################################
### PCA, IPCA, Randomized PCA under Different Number of Principle Components #######################################################################
####################################################################################################################################################
for n_components in (2, 10, 154):   # Conduct principle component analysis with different number of PCs
    
    print('n_components = ', n_components)
    
    regular_pca = PCA(n_components=n_components)        # Regular PCA with different number of PCS
    inc_pca = IncrementalPCA(n_components=n_components, batch_size=500)     # Incremental PCA with different number of PCs
    rnd_pca = PCA(n_components=n_components, random_state=42, svd_solver='randomized')      # Randomized PCA with different number of PCs

    for pca in (regular_pca, inc_pca, rnd_pca):

        # Measure the time taken for principle component analysis
        t1 = time.time()    # Measure time before PCA
        pca.fit(X_train)    # Acquire principle components with current MNIST training set
        t2 = time.time()    # Measure time after PCA

        print('     {} : {:.1f} seconds'.format(pca.__class__.__name__, t2-t1))

####################################################################################################################################################
### PCA vs Randomized PCA under Different Number of Samples ########################################################################################
####################################################################################################################################################
times_rpca = []     # Time measurements with Randomized PCA
times_pca = []      # Time measurements with PCA
sizes = [1000, 10000, 20000, 30000, 40000, 50000, 70000, 100000, 200000, 500000]    # Number of samples

for n_samples in sizes:
    
    X = np.random.randn(n_samples, 5)       # Prepare the random dataset with 5 features

    pca = PCA(n_components=2, svd_solver='randomized', random_state=42)     # Randomized PCA object for producing 2 principle components
    t1 = time.time()            # Measure time before PCA
    pca.fit(X)                  # Acquire principle components with current random dataset with 5 features
    t2 = time.time()            # Measure time after PCA
    times_rpca.append(t2-t1)    # Save time measurements of Randomized PCA

    pca = PCA(n_components=2)   # PCA object for producing 2 principle components 
    t1 = time.time()            # Measure the time before PCA
    pca.fit(X)                  # Acquire principle components with current random dataset with 5 features
    t2 = time.time()            # Measure the time after PCA
    times_pca.append(t2-t1)     # Save time measurements of PCA

# Plot the time complexity of PCA and Randomized PCA
plt.plot(sizes, times_rpca, "b-o", label="RPCA")
plt.plot(sizes, times_pca, "r-s", label="PCA")
plt.xlabel("n_samples")
plt.ylabel("Training time")
plt.legend(loc="upper left")
plt.title("PCA and Randomized PCA time complexity ")
plt.show()

####################################################################################################################################################
### PCA vs Randomized PCA under Different Number of Features #######################################################################################
####################################################################################################################################################
times_rpca = []     # Time measurements with Randomized PCA
times_pca = []      # Time measurements with PCA
sizes = [1000, 2000, 3000, 4000, 5000, 6000]        # Number of features

for n_features in sizes:
    
    X = np.random.randn(2000, n_features)       # Prepare the random dataset with different number of features
    
    pca = PCA(n_components=2, random_state=42, svd_solver='randomized')     # Randomized PCA object for producing 2 principle components
    t1 = time.time()                # Measure time before PCA
    pca.fit(X)                      # Acquire principle components with current random dataset with different number of features
    t2 = time.time()                # Measure time after PCA
    times_rpca.append(t2-t1)        # Save time measurements of Randomized PCA

    pca = PCA(n_components=2)       # PCA object for producing 2 principle components
    t1 = time.time()                # Measure the time before PCA
    pca.fit(X)                      # Acquire principle components with current random dataset with different number of features
    t2 = time.time()                # Measure the time after PCA
    times_pca.append(t2-t1)         # Save time measurements of PCA

# Plot the time complexity of PCA and Randomized PCA
plt.plot(sizes, times_rpca, "b-o", label="RPCA")
plt.plot(sizes, times_pca, "r-s", label="PCA")
plt.xlabel("n_features")
plt.ylabel("Training time")
plt.legend(loc="upper left")
plt.title("PCA and Randomized PCA time complexity ")
plt.show()