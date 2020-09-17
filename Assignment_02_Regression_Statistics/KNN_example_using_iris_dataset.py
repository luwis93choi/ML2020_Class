import numpy as np
import matplotlib as plt
import pandas as pd
import mglearn
import scipy as sp
import sklearn

from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

### Data Analysis Phase ###

iris_dataset = load_iris()

print('iris_dataset key : \n', iris_dataset.keys())

print(iris_dataset['DESCR'][:193] + '\n...')

print('Target Name : ', iris_dataset['target_names'])

print('Feature Names : ', iris_dataset['feature_names'])

print('Data Type : ', type(iris_dataset['data']))

print('Data Size : ', iris_dataset['data'].shape)

print('First 5 rows of the dataset : \n', iris_dataset['data'][:5])

print('Target Type : ', type(iris_dataset['target']))

print('Target Size : ', iris_dataset['target'].shape)

print('Target : \n', iris_dataset['target'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print('X_train Size : ', X_train.shape)
print('y_train Size : ', y_train.shape)

print('X_test Size : ', X_test.shape)
print('y_test Size : ', y_test.shape)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins' : 20}, s=60, alpha=0.8, cmap=mglearn.cm3)

plt.show()

### Classification using KNN (K-Nearest Neighbors) ###

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print('X_new.shape : ', X_new.shape)

prediction = knn.predict(X_new)
print('Prediction : ', prediction)
print('Prediction Target Name : ', iris_dataset['target_names'][prediction])

y_pred = knn.predict(X_test)
print('Predictions regarding Test Set : \n', y_pred)

print('Test Set Prediction Accuracy : {:.3f}'.format(np.mean(y_pred==y_test)))