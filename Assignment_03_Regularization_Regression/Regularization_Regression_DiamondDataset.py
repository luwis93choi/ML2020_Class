import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso, ElasticNetCV
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

### Dataset Preparation Phase ###
# Prepare the dataset and split it into train set and test set
X_train, X_test, y_train, y_test = train_test_split(features_X, target_y, test_size=0.25, random_state=101)
parameters = {'alpha': np.concatenate((np.arange(0.1,2,0.1), np.arange(2, 5, 0.5), np.arange(5, 100, 1)))}

### Regression Phase ###

# [Linear Regression]
# Apply Linear Regression Model for reference data
linear = linear_model.LinearRegression()

linear.fit(X_train, y_train)

y_predict = linear.predict(X_test)

print('----------------------------------')
print('Linear Score : ', linear.score(X_test, y_test))
print('Linear MSE : ', mean_squared_error(y_test, y_predict))

plt.subplot(2, 2, 1)
plt.title('Diamond Price Prediction using Linear Regression \n (R^2 Score : {:2f} / MSE : {:2f})'.format(linear.score(X_test, y_test), mean_squared_error(y_test, y_predict)))
plt.grid()
plt.plot(range(30000), range(30000), color='red', linestyle='dashed')
plt.xlim(0, 30000)
plt.ylim(0, 30000)
plt.scatter(y_test, y_predict)

# [Ridge = RSS + L2 Penalty]
# Since the correlation matrix shows multi-colinearity between features, 
# Ridge is used as regularization method to suppress the multi-colinearity between features.
ridge = linear_model.Ridge()

gridridge = GridSearchCV(ridge, parameters, scoring='r2')

gridridge.fit(X_train, y_train)

y_predict = gridridge.predict(X_test)

print('----------------------------------')
print('Ridge Best Parameters : ', gridridge.best_params_)
print('Ridge Score : ', gridridge.score(X_test, y_test))
print('Ridge MSE : ', mean_squared_error(y_test, y_predict))

plt.subplot(2, 2, 2)
plt.title('Diamond Price Prediction using Ridge \n (R^2 Score : {:2f} / MSE : {:2f})'.format(gridridge.score(X_test, y_test), mean_squared_error(y_test, y_predict)))
plt.grid()
plt.plot(range(30000), range(30000), color='red', linestyle='dashed')
plt.xlim(0, 30000)
plt.ylim(0, 30000)
plt.scatter(y_test, y_predict, color='orange')

# [Lasso = RSS + L1 Penalty]
# Lasso is used to further suppress the multi-colinearity among features
# By zeroing unvialbe features, Lasso can be used for feature selection.
lasso = linear_model.Lasso()

gridlasso = GridSearchCV(lasso, parameters, scoring='r2')

gridlasso.fit(X_train, y_train)

y_predict = gridlasso.predict(X_test)

print('----------------------------------')
print('Lasso Best Parameters : ', gridlasso.best_params_)
print('Lasso Score : ', gridlasso.score(X_test, y_test))
print('Lasso MSE : ', mean_squared_error(y_test, y_predict))

plt.subplot(2, 2, 3)
plt.title('Diamond Price Prediction using Lasso \n (R^2 Score : {:2f} / MSE : {:2f})'.format(gridlasso.score(X_test, y_test), mean_squared_error(y_test, y_predict)))
plt.grid()
plt.plot(range(30000), range(30000), color='red', linestyle='dashed')
plt.xlim(0, 30000)
plt.ylim(0, 30000)
plt.scatter(y_test, y_predict, color='black')

# [ElasticNet = Ridge + Lasso]
# ElasticNet is the combinatio of Ridge and Lasso.
# ElasticNet can balance between Ridge and Lasso. This can produce more generalized regression model for the dataset.
elasticNet = linear_model.ElasticNetCV(cv=5, random_state=12, l1_ratio=np.arange(0, 1, 0.01), alphas=np.arange(0.1, 100, 0.1))

elasticNet.fit(X_train, y_train)

y_predict = elasticNet.predict(X_test)

L1_ratio = elasticNet.l1_ratio_

print('----------------------------------')
print('ElasticNet')
print('L1 penalty ratio : ', L1_ratio)
print('ElasticNet Score : ', elasticNet.score(X_test, y_test))
print('ElasticNet MSE : ', mean_squared_error(y_test, y_predict))

plt.subplot(2, 2, 4)
plt.title('Diamond Price Prediction using ElasticNet \n (L1 Penalty : {:2f} / L2 Penalty : {:2f}) \n (R^2 Score : {:2f} / MSE : {:2f})'.format(elasticNet.l1_ratio_, 1 - elasticNet.l1_ratio_, elasticNet.score(X_test, y_test), mean_squared_error(y_test, y_predict)))
plt.grid()
plt.plot(range(30000), range(30000), color='red', linestyle='dashed')
plt.xlim(0, 30000)
plt.ylim(0, 30000)
plt.scatter(y_test, y_predict, color='green')
plt.show()

### Coefficient changes (Feature weight value changes) for parameter changes
alphaRidge = parameters['alpha']
alphaLasso = parameters['alpha']
alphaElasticNet = parameters['alpha']

coefficient_Ridge = []
coefficient_Lasso = []
coefficient_ElasticNet = []

# Prepare coefficient changes of Ridge
for alpha in alphaRidge:
    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefficient_Ridge.append(ridge.coef_[0])

# Prepare coefficient changes of Lasso
for alpha in alphaLasso:
    lasso = linear_model.Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    coefficient_Lasso.append(lasso.coef_)

# Prepare coefficient changes of ElasticNet
for alpha in alphaElasticNet:
    elasticNet = linear_model.ElasticNetCV(cv=5, random_state=12, l1_ratio=L1_ratio, alphas=[alpha])
    elasticNet.fit(X_train, y_train)
    coefficient_ElasticNet.append(elasticNet.coef_)

plt.cla
plt.subplot(1, 3, 1)
plt.plot(alphaRidge, coefficient_Ridge)
plt.title('Ridge Coefficients')
plt.xlabel('alpha')
plt.ylabel('coefficients')

plt.subplot(1, 3, 2)
plt.plot(alphaLasso, coefficient_Lasso)
plt.title('Lasso Coefficients')
plt.xlabel('alpha')
plt.ylabel('coefficients')

plt.subplot(1, 3, 3)
plt.plot(alphaElasticNet, coefficient_ElasticNet)
plt.title('ElasticNet Coefficients \n (L1 Penalty : {:2f} / L2 Penalty : {:2f})'.format(L1_ratio, 1-L1_ratio))
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.show()