import numpy as np
import matplotlib.pyplot as plt
import sklearn
import mglearn

from sklearn.datasets import fetch_lfw_people

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

####################################################################################################################################################
### Dataset Preparation and Analysis ###############################################################################################################
####################################################################################################################################################
# Load Labeled Faces in the Wild (LFW) people dataset 
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

image_shape = people.images[0].shape    # Acquire the shape of the image

# Display 10 examples from the dataset
fix, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks' : (), 'yticks' : ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

plt.show()

print('people.images.shape : {}'.format(people.images.shape))       # Print the shape of the image
print('Number of classes : {}'.format(len(people.target_names)))    # Print the number of classes in the dataset

# Count the number of data in each class
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print('{0:25} {1:3}'.format(name, count), end=' ')
    if (i + 1) % 3 == 0:
        print()

# Collect the images with matching target names
# Filter out the images with mismatched names
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]        # Use the image data with matching target names
y_people = people.target[mask]      # Use the target names that match the images

X_people = X_people / 255   # Scale the grayscale image values between 0 and 1

# Split the dataset between training dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

####################################################################################################################################################
### KNN Classification with Original Dataset #######################################################################################################
####################################################################################################################################################
knn = KNeighborsClassifier(n_neighbors=1)   # Prepare KNN classifier with 1 neighbor
knn.fit(X_train, y_train)                   # Train KNN classifier with training dataset

print()
print('Test set score of 1-nn : {:.2f}'.format(knn.score(X_test, y_test)))  # Print the accuracy of KNN classifier

# Plot PCA whitening process
mglearn.plots.plot_pca_whitening()
plt.show()

####################################################################################################################################################
### Principle Componeny Analysis of LFW dataset ####################################################################################################
####################################################################################################################################################
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)   # PCA object that produces 100 principle components for LFW dataset
X_train_pca = pca.transform(X_train)    # Re-Organize/Transform training dataset based on principle components
X_test_pca = pca.transform(X_test)      # Re-Organize/Transform test dataset based on principle components

print('X_train_pca.shape : {}'.format(X_train_pca.shape))   # Print the shape of PCA-based data

####################################################################################################################################################
### KNN Classification of PCA-based Dataset ########################################################################################################
####################################################################################################################################################
knn = KNeighborsClassifier(n_neighbors=1)   # Prepare KNN classifier with 1 neighbor
knn.fit(X_train_pca, y_train)               # Train KNN classifier with PCA-based training dataset

print('Test set accuracy : {:.2f}'.format(knn.score(X_test_pca, y_test)))   # Print the accuracy of KNN classifier with PCA-based test dataset

print('pca.components_.shape : {}'.format(pca.components_.shape))   # Print the shape of principle components

# Display the result images composed of different number of principle components
fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("{}. component".format((i + 1)))

plt.show()

# Display the image construction results under different number of principle components
mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)

plt.show()

# Scatter plot of PCA-based dataset
mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

plt.show()