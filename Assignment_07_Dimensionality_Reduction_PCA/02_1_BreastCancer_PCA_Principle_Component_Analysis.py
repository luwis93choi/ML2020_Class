import numpy as np
import matplotlib.pyplot as plt
import sklearn
import mglearn

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Display PCA computation process
mglearn.plots.plot_pca_illustration()
plt.show()

####################################################################################################################################################
### Dataset Preparation and Analysis ###############################################################################################################
####################################################################################################################################################
# Load breast cancer dataset
cancer = load_breast_cancer()

fig, axes = plt.subplots(6, 5, figsize=(10, 20))

malignant = cancer.data[cancer.target == 0]     # Data classified as malignant
benign = cancer.data[cancer.target == 1]        # Data classified as benign

# Plot the histogram of each feature in dataset
ax = axes.ravel()
for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i], fontsize=10)
    ax[i].set_yticks(())
    ax[0].set_ylabel("Frequency", fontsize=10)
    ax[0].legend(["malignant", "benign"], loc="best")

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.5)
plt.show()

# Dataset standardization
scaler = StandardScaler()                   # Prepare standard scaler
scaler.fit(cancer.data)                     # Acquire mean and standard deviation of the dataset
X_scaled = scaler.transform(cancer.data)    # Standardize the dataset

####################################################################################################################################################
### Principle Component Analysis of Breast Cancer Dataset ##########################################################################################
####################################################################################################################################################
pca = PCA(n_components=2)   # PCA object that produces 2 principle components
pca.fit(X_scaled)           # Fit PCA object with standardized dataset

X_pca = pca.transform(X_scaled)     # Re-Organize/Transform the dataset based on principle components

print('Original shape : {}'.format(str(X_scaled.shape)))    # Print the shape of original data
print('Reduced shape : {}'.format(str(X_pca.shape)))        # Print the shape of PCA-based data

####################################################################################################################################################
### Result Plotting ################################################################################################################################
####################################################################################################################################################

# Plot Re-Organized dataset
plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:,0], X_pca[:,1],cancer.target)
plt.legend(["malignant", "benign"], loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")    
plt.show()      

print('PCA shape : ', pca.components_.shape)    # Print the shape of principle components

print('PCA components : ', pca.components_)     # Print principle components

# Plot the principle components
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
plt.show()