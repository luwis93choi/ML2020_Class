import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

np.random.seed(42)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = '.'
CHAPTER_ID = 'dim_reduction'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Save Image ', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import warnings
warnings.filterwarnings(action='ignore', message='^internal gelsd')

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

### SVD-based Dimensionality Reduction ###
X_centered = X - X.mean(axis=0)     # Standardize the dataset

U, s, Vt = np.linalg.svd(X_centered)    # SVD = Computation for Acquiring Principle Components (Eigen Vectors)
# U = Left Singlular Vector = Product of A * A^T
# Vt = Right Singular Vector = Product of A^T * A = Principle Components from SVD
# s = Eigen Values for A * A^T and A^T * A

c1 = Vt.T[:, 0]     # 1st Principle Component from SVD
c2 = Vt.T[:, 1]     # 2nd Principle Component from SVD

m, n = X.shape

S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)

print(np.allclose(X_centered, U.dot(S).dot(Vt)))    # Verify if the results of SVD and original dataset data are similar to each other
                                                    # U * S * Vt = Reprojection back to original feature space

W2 = Vt.T[:, :2]            # Choose 1st and 2nd PC
X2D_using_svd = X_centered.dot(W2)    # Project standardized dataset onto 1st and 2nd PC
                                      # --> Re-Organized the dataset with 1st and 2nd PCs as new features

### PCA-based Dimensionality Reduction ###
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

print(X2D[:5])
print(X2D_using_svd[:5])

print(np.allclose(X2D, -X2D_using_svd))