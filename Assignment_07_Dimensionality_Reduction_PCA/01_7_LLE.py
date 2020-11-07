import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

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

####################################################################################################################################################
### Prepare 3D Swiss Roll Dataset ##################################################################################################################
####################################################################################################################################################
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)

####################################################################################################################################################
### LLE (Locally Linear Embedding) - NonLinear Dimensionality Reduction Technique / Manifold Learning ##############################################
####################################################################################################################################################

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)       
# LLE PCA object that produces 2 principle component from 10 neighbors around each data

X_reduced = lle.fit_transform(X)    # Re-Organize the dataset with principle components

# Plot the unrolled swiss roll that is produced by LLE
plt.title("Unrolled swiss roll using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)

save_fig("lle_unrolling_plot")
plt.show()