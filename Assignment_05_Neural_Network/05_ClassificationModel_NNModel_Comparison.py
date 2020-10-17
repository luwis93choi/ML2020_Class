#################################################################################
### Breast Cancer Classification using various Classifiers and Neural Network ###
#################################################################################

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics

from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

breast_cancer_dataset = load_breast_cancer()

print('[Breast Cancer Classification using various Classifiers and Neural Network] \n')

print('Breast Cancer Dataset Features')
print(breast_cancer_dataset.feature_names, '\n')

print('Breast Cancer Dataset Target : {} \n'.format(breast_cancer_dataset.target_names))

# Prepare original dataset
breast_cancer_data = pd.DataFrame(breast_cancer_dataset.data)
breast_cancer_label = pd.DataFrame(breast_cancer_dataset.target)

# Prepare standardized dataset using Standard Scaler
stdScaler = StandardScaler()
stdScaler.fit(breast_cancer_data)
breast_cancer_data_std = stdScaler.transform(breast_cancer_data)

# Disply dataset distribution in order to compare original dataset and standardized dataset
main_fig = plt.figure(figsize=[20, 8])
gs = main_fig.add_gridspec(2, 6)

fg_ax1 = main_fig.add_subplot(gs[0, 0:2])
fg_ax1.boxplot(breast_cancer_dataset.data)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax1.set_title('Original Breast Cancer Dataset\n(No Standardization or Normalization)', fontdict={'weight':'bold'})

fg_ax2 = main_fig.add_subplot(gs[0, 2:4])
fg_ax2.boxplot(breast_cancer_data_std)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax2.set_title('Standardized Breast Cancer Dataset', fontdict={'weight':'bold'})

### Coefficient Matrix Heatmap ###
##################################

# Dataset Group 
dataset_group = [{'name' : 'original', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None},
                 {'name' : 'standardized', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None}]

# Split original / standardized breast cancer dataset into 70% training set and 30% test set
(dataset_group[0]['X_train'], dataset_group[0]['X_test'], 
 dataset_group[0]['y_train'], dataset_group[0]['y_test']) = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.3, random_state=23)

(dataset_group[1]['X_train'], dataset_group[1]['X_test'], 
 dataset_group[1]['y_train'], dataset_group[1]['y_test']) = train_test_split(breast_cancer_data_std, breast_cancer_label, test_size=0.3, random_state=23)

accuracy = [{'name' : 'original', 'LinearR_acc' : None, 'LogR_acc' : None, 'maxKNN_acc' : None, 'GaussianNB_acc' : None, 'RandomForest_acc' : None, 
             'DecisionTree_acc' : None, 'SVM_acc' : None, 'perceptron_acc' : None, 'MLP_acc' : None},
            {'name' : 'standardized', 'LinearR_acc' : None, 'LogR_acc' : None, 'maxKNN_acc' : None, 'GaussianNB_acc' : None, 'RandomForest_acc' : None, 
             'DecisionTree_acc' : None, 'SVM_acc' : None, 'perceptron_acc' : None, 'MLP_acc' : None}]

### Classifier training & Prediction over all dataset groups ###
for dataset in dataset_group:

    ### Linear Regression ###
    LinearR_model = LinearRegression()
    LinearR_model.fit(dataset['X_train'], dataset['y_train'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['LinearR_acc'] = LinearR_model.score(dataset['X_test'], dataset['y_test'])

    ### Logistic Regression ###
    LogR_model = LogisticRegression()
    LogR_model.fit(dataset['X_train'], dataset['y_train'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['LogR_acc'] = LogR_model.score(dataset['X_test'], dataset['y_test'])

    ### KNN ###
    k_list = range(1, 101)
    knn_accuracy = []

    for k in k_list:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(dataset['X_train'], dataset['y_train'])

        prediction = knn_model.predict(dataset['X_test'])

        knn_accuracy.append(metrics.accuracy_score(prediction, dataset['y_test']))

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['maxKNN_acc'] = max(knn_accuracy)

    ### Gaussian Naive Bayes ###
    GaussiaNB_model = GaussianNB()
    GaussiaNB_model.fit(dataset['X_train'], dataset['y_train'])

    prediction = GaussiaNB_model.predict(dataset['X_test'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['GaussianNB_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])

    ### Random Forest ###
    RandomForest = RandomForestClassifier(n_estimators=100)
    RandomForest.fit(dataset['X_train'], dataset['y_train'])

    prediction = RandomForest.predict(dataset['X_test'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['RandomForest_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])

    ### Decision Tree ###
    DecisionTree = DecisionTreeClassifier()
    DecisionTree.fit(dataset['X_train'], dataset['y_train'])

    prediction = DecisionTree.predict(dataset['X_test'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['DecisionTree_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])

    ### Support Vector Machine ###
    SVM = svm.SVC(kernel='rbf', C=1, gamma=0.1)
    SVM.fit(dataset['X_train'], dataset['y_train'])

    prediction = SVM.predict(dataset['X_test'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['SVM_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])

    ### Perceptron ###
    perceptron = Perceptron(max_iter=40, eta0=0.001, tol=1e-3, random_state=23)
    perceptron.fit(dataset['X_train'], dataset['y_train'])

    prediction = perceptron.predict(dataset['X_test'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['perceptron_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])

    ### Multi-Layer Perceptron ###
    mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=23)
    mlp.fit(dataset['X_train'], dataset['y_train'])

    prediction = mlp.predict(dataset['X_test'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['MLP_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])

# Plot various classification accuracy for each type of datasets
models = ['Linear\nRegression', 'Logistic\nRegression', 'KNN', 'GaussianNB', 'Random\nForest', 'Decision\nTree', 'SVM', 'Perceptron', 'MLP\nLayer : [100, 100]']

width = 0.35
fg_ax4 = main_fig.add_subplot(gs[1, 0:6])
fg_ax4.bar(np.arange(len(models)) - width/4, [accuracy[0]['LinearR_acc'], accuracy[0]['LogR_acc'], accuracy[0]['maxKNN_acc'], 
                                              accuracy[0]['GaussianNB_acc'], accuracy[0]['RandomForest_acc'], accuracy[0]['DecisionTree_acc'], 
                                              accuracy[0]['SVM_acc'], accuracy[0]['perceptron_acc'], accuracy[0]['MLP_acc']], width=width/3, label='Original Dataset')
fg_ax4.bar(np.arange(len(models)) + width/4, [accuracy[1]['LinearR_acc'], accuracy[1]['LogR_acc'], accuracy[1]['maxKNN_acc'], 
                                              accuracy[1]['GaussianNB_acc'], accuracy[1]['RandomForest_acc'], accuracy[1]['DecisionTree_acc'], 
                                              accuracy[1]['SVM_acc'], accuracy[1]['perceptron_acc'], accuracy[1]['MLP_acc']], width=width/3, label='Standardized Dataset')
fg_ax4.set_title('Classification Result Comparison : Original / Standardized / Normalized', fontdict={'weight':'bold', 'size' : 15})
plt.xticks(np.arange(len(models)), models, fontweight='bold')
plt.xlim(-0.3, 10.1)
plt.ylim(0, 1.1)
plt.ylabel('Accuracy', fontweight='bold')
fg_ax4.legend(loc='upper right')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = '\n'.join((
    'Linear Regression (Original) : ' + "{:.4f}".format(accuracy[0]['LinearR_acc']),
    'Linear Regression (Standardized) : ' + "{:.4f}".format(accuracy[1]['LinearR_acc']),
    'Logistic Regression (Original) : ' + "{:.4f}".format(accuracy[0]['LogR_acc']),
    'Logistic Regression (Standardized) : ' + "{:.4f}".format(accuracy[1]['LogR_acc']),
    'KNN (Original) : ' + "{:.4f}".format(accuracy[0]['maxKNN_acc']), 
    'KNN (Standardized) : ' + "{:.4f}".format(accuracy[1]['maxKNN_acc']),
    'GaussianNB (Original) : ' + "{:.4f}".format(accuracy[0]['GaussianNB_acc']), 
    'GaussianNB (Standardized) : ' + "{:.4f}".format(accuracy[1]['GaussianNB_acc']),
    'Random Forest (Original) : ' + "{:.4f}".format(accuracy[0]['RandomForest_acc']),
    'Random Forest (Standardized) : ' + "{:.4f}".format(accuracy[1]['RandomForest_acc']),
    'Decision Tree (Original) : ' + "{:.4f}".format(accuracy[0]['DecisionTree_acc']),
    'Decision Tree (Standardized) : ' + "{:.4f}".format(accuracy[1]['DecisionTree_acc']),
    'SVM (Original) : ' + "{:.4f}".format(accuracy[0]['SVM_acc']),
    'SVM (Standardized) : ' + "{:.4f}".format(accuracy[1]['SVM_acc']),
    'Perceptron (Original) : ' + "{:.4f}".format(accuracy[0]['perceptron_acc']),
    'Perceptron (Standardized) : ' + "{:.4f}".format(accuracy[1]['perceptron_acc']),
    'MLP Layer : [100, 100] (Original) : ' + "{:.4f}".format(accuracy[0]['MLP_acc']),
    'MLP Layer : [100, 100] (Standardized) : ' + "{:.4f}".format(accuracy[1]['MLP_acc'])))

fg_ax4.text(0.85, 0.1, textstr, transform=fg_ax4.transAxes, fontsize=8, bbox=props)

plt.tight_layout()
plt.show()