import numpy as np
import pandas as pd
import random

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load Diamond Dataset
diamonds = pd.read_csv('diamonds.csv')
print('[First 5 rows of Diamond Dataset]')
print(diamonds.head())

### Preprocessing Phase ###

# Drop the column data with unnamed index
diamonds = diamonds.drop(['Unnamed: 0'], axis=1)
print('[First 5 rows of Diamond Dataset (Without Unnamed Data)]')
print(diamonds.head())

# Encode text feature values into numerical labels using Label Encoder
text_cateogrical_features = ['cut', 'color', 'clarity']
le = LabelEncoder()

for i in range(len(text_cateogrical_features)):
    numeric_label = le.fit_transform(diamonds[text_cateogrical_features[i]])
    diamonds[text_cateogrical_features[i]] = numeric_label

print('[First 5 rows of Diamond Dataset (Encoded Label)]')
print(diamonds.head())

print('[Number of NaN/NULL data]')
print(diamonds.isnull().sum())

# Dataset Scaling
# Since all the features are using different units, they need to be scaled/standardized into same units through rescaling
# Using different units can cause negative effects on the training results. 
# This is because under different units, each feature has relative effect on training process compared to others.
# In other words, certain features with high unit and smaller magnitude changes can be ignored when compared to the features with small unit and greater magnitude changes.
features_X = diamonds[['carat', 'depth', 'table', 'x', 'y', 'z', 'clarity', 'cut', 'color']]
target_y = diamonds[['price']]

scaler = StandardScaler()
scaler.fit(features_X)
features_X = scaler.transform(features_X)

### Dataset Analysis Phase ###

# If the correlation values between different features are above 0, it means there is multi-colinearity among features.
# With multi-colinearity between features, linear regression cannot produce an appropriate with optimal fitting.
# This requires 'Regularization (Shrinkage)' that produces the regression model with reduced multi-colinearity.

# Check Multi-Colinearity between features by calculating the correlation matrix of features
DIAMONDS_X = pd.DataFrame(features_X, 
                          columns=['carat', 'depth', 'table', 'x', 'y', 'z', 'clarity', 'cut', 'color'], 
                          index=range(len(features_X)))

correlation_matrix = DIAMONDS_X.corr()

# Display the correlation matrix as heatmap and determine the multi-colinearity between features
plt.figure(figsize=(12, 12))
plt.title('Correlation Matrix Heatmap of Diamond Dataset', pad=15, fontsize='x-large')
sns.heatmap(data=correlation_matrix, square=True, annot=True, cbar=True)
plt.show()

# Prepare the dataset and split it into train set and test set
X_train, X_test, y_train, y_test = train_test_split(features_X, target_y, test_size=0.25, random_state=101)
parameters = {'alpha': np.concatenate((np.arange(0.1,2,0.1), np.arange(2, 5, 0.5), np.arange(5, 26, 1)))}

lasso = linear_model.Lasso()
gridlasso = GridSearchCV(lasso, parameters, scoring='r2')
gridlasso.fit(X_train, y_train)
print('Lasso Best Parameters : ', gridlasso.best_params_)
print('Lasso Score : ', gridlasso.score(X_test, y_test))
print('Lasso MSE : ', mean_squared_error(y_test, gridlasso.predict(X_test)))