import mglearn
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sklearn

mpl.rc('font', family='DejaVu Sans')
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(['Class 0', 'Class 1'], loc=4)
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
print('X.shape', X.shape)

plt.show()

plt.cla()

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('Feature')
plt.ylabel('Target')

plt.show()

######################################################################

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print('cancer.keys() : \n', cancer.keys())

print('Breast cancer dataset shape : ', cancer.data.shape)

print('# of Samples per Class \n', {n : v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})

print('Feature Name : \n', cancer.feature_names)

######################################################################

mglearn.plots.plot_linear_regression_wave()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print('Correlation (Linearity) : ', np.corrcoef(np.array(X.T)[0], y)[1][0])

print('X shape : ', X.shape)
print('Y shape : ', y.shape)

print('X_train : ', X_train)
print('y_train : ', y_train)

print('X_test : ', X_test)
print('y_test : ', y_test)

lr = LinearRegression().fit(X_train, y_train)
print('lr.coef_ : ', lr.coef_)
print('lr.intercept_ : ', lr.intercept_)

print('Train set score : {:.2f}'.format(lr.score(X_train, y_train)))
print('Test set score : {:.2f}'.format(lr.score(X_test, y_test)))

x_line = np.linspace(-3, 3)
y_line = lr.coef_[0] * x_line + lr.intercept_

plt.subplot(1, 3, 1)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid()
plt.scatter(X, y)
plt.plot(x_line, y_line, color='black')

plt.subplot(1, 3, 2)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid()
plt.scatter(X_train, y_train)
plt.plot(x_line, y_line, color='black')

plt.subplot(1, 3, 3)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid()
plt.scatter(X_test, y_test)
plt.plot(x_line, y_line, color='black')

plt.show()

######################################################################

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print('X shape : ', X.shape)
print('Y shape : ', y.shape)

print('Train set score : {:.2f}'.format(lr.score(X_train, y_train)))
print('Test set score : {:.2f}'.format(lr.score(X_test, y_test)))

