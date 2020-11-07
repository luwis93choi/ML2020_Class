import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll

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

### Prepare 3D Swiss Roll Dataset
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

### Kernel PCA with different types of kernel
lin_pca = KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True)    # Linear PCA with Linear Kernel
rbf_pca = KernelPCA(n_components=2, kernel='rbf', fit_inverse_transform=True)       # RBF PCA with Radial Basis Functio Kernel
sig_pca = KernelPCA(n_components=2, kernel='sigmoid', fit_inverse_transform=True)   # Sigmoid PCA with Sigmoid Kernel

plt.figure(figsize=(11, 4))

for subplot, pca, title in ((131, lin_pca, 'Linear Kernel'), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):

    