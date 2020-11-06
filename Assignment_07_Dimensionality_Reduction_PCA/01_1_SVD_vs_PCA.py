import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from mpl_toolkits.mplot3d import Axes3D

####################################################################################################################################################
### Preparation ####################################################################################################################################
####################################################################################################################################################
np.random.seed(42)

mpl.rc('axes', labelsize=14)        # Define the size of axes
mpl.rc('xtick', labelsize=12)       # Define x axis tick size
mpl.rc('ytick', labelsize=12)       # Define y axis tick size

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

import warnings
warnings.filterwarnings(action='ignore', message='^internal gelsd')

####################################################################################################################################################
### Random Dataset Generation ######################################################################################################################
####################################################################################################################################################

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

####################################################################################################################################################
### Dimensionality Reduction #######################################################################################################################
####################################################################################################################################################

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
                                                    # U * S * Vt = Composition back to original dataset

W2 = Vt.T[:, :2]            # Choose 1st and 2nd PC
X2D_using_svd = X_centered.dot(W2)    # Project standardized dataset onto 1st and 2nd PC
                                      # Produce projected 
                                      # --> Re-Organized the dataset with 1st and 2nd PCs as new features

### PCA-based Dimensionality Reduction ###
pca = PCA(n_components=2)       # PCA object for producing top 2 principle components for the dataset
                                # --> Re-Organized the dataset on 2D dimension with 1st and 2nd PCs as new features

X2D = pca.fit_transform(X)      # Produce Re-Organized dataset (sklearn API includes standardization)

print(X2D[:5])
print(X2D_using_svd[:5])

print(np.allclose(X2D, -X2D_using_svd)) # Verify if the results of PCA and SVD are similar to each other

X3D_inv = pca.inverse_transform(X2D)    # Recovered dataset : Reprojection back to original feature space

print(np.allclose(X3D_inv, X))          # Verify if the original dataset and recovered dataset are similar or equal within 1e-8 error tolerance
                                        # --> False, because dataset loses some information during reprojection between 3D and 2D

print(np.mean(np.sum(np.square(X3D_inv - X), axis=1)))  # Print error rate between original dataset and recovered datset

X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])    # Reprojection back to original space using SVD results

print(np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean_))  # Verify if SVD-based recovered dataset and PCA-bsed recovered dataset are similar to each other
                                                            # Subtract mean of dataset since sklearn PCA automatically adds up mean in its API

print(pca.components_)      # Print top 2 principle components from PCA

print(Vt[:2])               # Print top 2 principle componets from SVD

print(pca.explained_variance_ratio_)      # Print the varaince of each principle component from PCA

print(1 - pca.explained_variance_ratio_.sum())  # Print the variance loss of PCA

print(np.square(s) / np.square(s).sum())  # Print the variance of each principle component from SVD

####################################################################################################################################################
### Dimensionality Reduction Plotting ##############################################################################################################
####################################################################################################################################################

# 3D Arrow class for describing reprojection
class Arrow3D(FancyArrowPatch):
    # Initialize 3D Arrow object
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    # Draw function for 3D Arrow
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]    # Axis range for 3D dimensions

x1s = np.linspace(axes[0], axes[1], 10)     # Line Space between Axis 1
x2s = np.linspace(axes[2], axes[3], 10)     # Line Space between Axis 2
x1, x2 = np.meshgrid(x1s, x2s)

C = pca.components_     # Load top 2 principle components
R = C.T.dot(C)          # Projection between PCs to create a plane
z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])   # Plane made of top 2 principle components

fig = plt.figure(figsize=(6, 3.8))
ax = fig.add_subplot(111, projection='3d')

X3D_above = X[X[:, 2] > X3D_inv[:, 2]]      # Collect values above new 2D plane
X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]     # Collect values below new 2D plane

ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)

ax.plot_surface(x1, x2, z, alpha=0.2, color="k")    # Draw 2D plane made of top 2 principle components from PCA
np.linalg.norm(C, axis=0)
ax.add_artist(Arrow3D([0, C[0, 0]],[0, C[0, 1]],[0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))    # Draw the arrow for principle component
ax.add_artist(Arrow3D([0, C[1, 0]],[0, C[1, 1]],[0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))    # Draw the arrow for principle component
ax.plot([0], [0], [0], "k.")

# Draw line between original dataset and reprojected data
for i in range(m):
    if X[i, 2] > X3D_inv[i, 2]:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")
    else:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-", color="#505050")

# Plot original dataset in 3D space and reprojected dataset in 2D plane
ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
ax.set_xlabel("$x_1$", fontsize=18, labelpad=10)
ax.set_ylabel("$x_2$", fontsize=18, labelpad=10)
ax.set_zlabel("$x_3$", fontsize=18, labelpad=10)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("dataset_3d_plot")     # Save the plotting result
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

# Plot reprojected dataset in 2D plane made of principle components as new features
ax.plot(X2D[:, 0], X2D[:, 1], "k+")
ax.plot(X2D[:, 0], X2D[:, 1], "k.")
ax.plot([0], [0], "ko")
ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')   # Draw the arrow for principle component
ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')   # Draw the arrow for principle component
ax.set_xlabel("$z_1$", fontsize=18)
ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
ax.axis([-1.5, 1.3, -1.2, 1.2])
ax.grid(True)
save_fig("dataset_2d_plot")

plt.show()

####################################################################################################################################################
