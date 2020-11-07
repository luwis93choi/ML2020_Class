import numpy as np
import matplotlib.pyplot as plt
import sklearn
import mglearn

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

####################################################################################################################################################
### Load Digits Dataset ############################################################################################################################
####################################################################################################################################################
digits = load_digits()

# Plot 10 examples from the dataset
fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks':(), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)

print(digits.images.shape)  # Print the shape of image

####################################################################################################################################################
### Principle Component Analysis of Digits Dataset #################################################################################################
####################################################################################################################################################
pca = PCA(n_components=2)   # PCA object that produces 2 principle components
pca.fit(digits.data)        # Produce 2 principle components with Digits dataset

digits_pca = pca.transform(digits.data)     # Re-Organize/Transform Digits dataset with principle components

# Plot the distribution of PCA-based Digits dataset
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())    # Set the limit of X axis
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())    # Set the limit of Y axis

# For each data in Digits dataset, plot the data based on the values of 1st and 2nd principle components
for i in range(len(digits.data)):
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})

plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()

####################################################################################################################################################
### t-SNE (t-Stochastic Neighbor Embedding) of Digits Dataset ######################################################################################
####################################################################################################################################################
tsne = TSNE(random_state=42)    # t-SNE object that produces t-SNE features

digits_tsne = tsne.fit_transform(digits.data)   # Re-Organize/Transform the dataset with t-SNE

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)  # Set the limit of X axis
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)  # Set the limit of Y axis

# For each data in Digits dataset, plot the data based on the values of t-SNE features
for i in range(len(digits.data)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
plt.show()