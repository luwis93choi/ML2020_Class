import numpy as np
import os
import urllib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import imread

import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

### Prepare an image ###
images_path = os.path.join('./', "dataset")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/rickiepark/handson-ml2/master/"
filename = "ladybug.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/unsupervised_learning/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

image = imread(os.path.join(images_path, filename))
print(image.shape)

X = image.reshape(-1, 3)    # Reshape 3D image data structure into 2D data list (list of X Y Z coordinates)
print(X.shape)

segmented_imgs = []     # List for saving segmented images

# Conduct K-Means clustering with a different number of clusters
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)      # Cluster the data
    
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]     # Each point in the image will have its label that corresponds to its cluster.
                                                                # Map each point in the image to the coordinates of its cluster/label 
                                                                # => Segment the image by mapping each image point to the cluster

    segmented_imgs.append(segmented_img.reshape(image.shape))   # Reshape back to original 3D image structure for display

### Show the results of image segmentation using K-Means with different number of clusters
plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')

plt.show()