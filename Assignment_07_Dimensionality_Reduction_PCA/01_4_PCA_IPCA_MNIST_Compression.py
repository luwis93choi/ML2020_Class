import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

# Define the directory for saving the plotting results
PROJECT_ROOT_DIR = '.'
CHAPTER_ID = 'dim_reduction'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# Function of saving the plotting results
def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Save Image ', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# MNIST dataset display function
def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    
####################################################################################################################################################
### Prepare MNIST dataset ##########################################################################################################################
####################################################################################################################################################
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

X = mnist['data']       # Load data from MNIST dataset
y = mnist['target']     # Load target values from MNIST dataset

X_train, X_test, y_train, y_test = train_test_split(X, y)   # Split the dataset between training and test

####################################################################################################################################################
### PCA (Principle Component Analysis) #############################################################################################################
####################################################################################################################################################
pca = PCA()             # PCA object for producing principle components for the dataset
pca.fit(X_train)        # Produce Re-Organized dataset (sklearn API includes standardization)
cumsum = np.cumsum(pca.explained_variance_ratio_)   # Cumulative sum of PC variance
d = np.argmax(cumsum >= 0.95) + 1                   # The number of PCs for explaining 95% of the Re-Organized dataset

print(d)    # Print the number of PCs for explaining 95% of the Re-Organized dataset

plt.figure(figsize=(6,4))
plt.plot(cumsum, linewidth=3)               # Plot the cumulative sum of PC variance
plt.axis([0, 400, 0, 1])
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")

# Indicate the point of explaining 95% of the Re-Organized dataset
plt.plot([d, d], [0, 0.95], "k:")           
plt.plot([0, d], [0.95, 0.95], "k:")
plt.plot(d, 0.95, "ko")
plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
             arrowprops=dict(arrowstyle="->"), fontsize=16)     # Indicate the point of explaining 95% of the Re-Organized dataset

plt.grid(True)
save_fig("explained_variance_plot")
plt.show()

pca = PCA(n_components=0.95)                    # PCA object for producing principle components that explain 95% of the dataset
X_reduced = pca.fit_transform(X_train)          # Produce Re-Organized dataset (sklearn API includes standardization)

print(np.sum(pca.explained_variance_ratio_))    # Print the variance ratio of current principle components

pca = PCA(n_components=154)                     # PCA object for producing 154 principle components
X_reduced = pca.fit_transform(X_train)          # Produce Re-Organized dataset (sklearn API includes standardization)
X_recovered = pca.inverse_transform(X_reduced)  # Reproject Re-Organized dataset back to original feature space

X_reduced_pca = X_reduced   # Save PCA-based Re-Organized dataset

# Comparison between original image dataset and PCA image dataset 
plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])                # Show training dataset of MNIST
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])            # Show PCA-reprojected training dataset 
plt.title("Compressed", fontsize=16)

save_fig("mnist_compression_plot")
plt.show()

####################################################################################################################################################
### IPCA (Incremental Principle Component Analysis) ################################################################################################
####################################################################################################################################################
n_batches = 100

inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches):
    print('.', end='')
    inc_pca.partial_fit(X_batch)
print()

X_reduced_inc_pca = inc_pca.transform(X_train)

X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced_inc_pca)

# Comparison between PCA image dataset and IPCA image dataset 
plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.subplot(122)
plot_digits(X_recovered_inc_pca[::2100])
plt.tight_layout()
plt.show()

####################################################################################################################################################
### Comparison between PCA and IPCA ################################################################################################################
####################################################################################################################################################

print(np.allclose(pca.mean_, inc_pca.mean_))

print(np.allclose(X_reduced_pca, X_reduced_inc_pca))
