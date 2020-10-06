#############################################################################
### Breaset Cancer Classification using Standardization and Normalization ###
#############################################################################

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics

from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

breast_cancer_dataset = load_breast_cancer()

print('Breast Cancer Dataset Key : ', breast_cancer_dataset.keys())

print('Breast Cancer Dataset Target : ', breast_cancer_dataset.target_names)

breast_cancer_data = pd.DataFrame(breast_cancer_dataset.data)
breast_cancer_label = pd.DataFrame(breast_cancer_dataset.target)

print(breast_cancer_data.head())

print(breast_cancer_label.head())

print(breast_cancer_data.describe())

# Standardization : Standardize the dataset into z-score
mean = breast_cancer_dataset.data.mean(axis=0)
standard_deviation = breast_cancer_dataset.data.std(axis=0)

standardized_breast_cancer_data = (breast_cancer_dataset.data - mean) / standard_deviation

# Normalization : Normalizae the dataset as the value between 0 and 1 using min-max normalization
dataset_min = breast_cancer_dataset.data.min(axis=0)
dataset_max = breast_cancer_dataset.data.max(axis=0)
normalized_breast_cancer_data = (breast_cancer_dataset.data - dataset_min) / (dataset_max - dataset_min)

# Plotting comparison between original / standardized / normalized breast cancer dataset
main_fig = plt.figure(figsize=[20, 8])
gs = main_fig.add_gridspec(2, 6)

fg_ax1 = main_fig.add_subplot(gs[0, 0:2])
fg_ax1.boxplot(breast_cancer_dataset.data)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax1.set_title('Original Breast Cancer Dataset\n(No Standardization or Normalization)', fontdict={'weight':'bold'})

fg_ax2 = main_fig.add_subplot(gs[0, 2:4])
fg_ax2.boxplot(standardized_breast_cancer_data)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax2.set_title('Standardized Breast Cancer Dataset', fontdict={'weight':'bold'})

fg_ax3 = main_fig.add_subplot(gs[0, 4:6])
fg_ax3.boxplot(normalized_breast_cancer_data)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax3.set_title('Normalized Breast Cancer Dataset', fontdict={'weight':'bold'})

# Split original / standardized / normalized breast cancer dataset into 70% training set and 30% test set
original_train_data, original_test_data, original_train_label, original_test_label = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.3, random_state=23)

standardized_train_data, standardized_test_data, standardized_train_label, standardized_test_label = train_test_split(standardized_breast_cancer_data, breast_cancer_label, test_size=0.3, random_state=23)

normalized_train_data, normalized_test_data, normalized_train_label, normalized_test_label = train_test_split(normalized_breast_cancer_data, breast_cancer_label, test_size=0.3, random_state=23)

# Prepare dictionary for each dataset and accuracy for iterating the datasets with various classification methods
dataset_group = [{'name' : 'original', 'train_data' : original_train_data, 'train_label' : original_train_label, 'test_data' : original_test_data, 'test_label' : original_test_label},
                 {'name' : 'standardized', 'train_data' : standardized_train_data, 'train_label' : standardized_train_label, 'test_data' : standardized_test_data, 'test_label' : standardized_test_label},
                 {'name' : 'normalized', 'train_data' : normalized_train_data, 'train_label' : normalized_train_label, 'test_data' : normalized_test_data, 'test_label' : normalized_test_label}]

accuracy = [{'name' : 'original', 'LogR_acc' : None, 'maxKNN_acc' : None, 'GaussianNB_acc' : None, 'RandomForest_acc' : None, 'DecisionTree_acc' : None, 'SVM_acc' : None},
            {'name' : 'standardized', 'LogR_acc' : None, 'maxKNN_acc' : None, 'GaussianNB_acc' : None, 'RandomForest_acc' : None, 'DecisionTree_acc' : None, 'SVM_acc' : None},
            {'name' : 'normalized', 'LogR_acc' : None, 'maxKNN_acc' : None, 'GaussianNB_acc' : None, 'RandomForest_acc' : None, 'DecisionTree_acc' : None, 'SVM_acc' : None}]

for dataset in dataset_group:

    ### Logistic Regression ###
    LogR_model = LogisticRegression()
    LogR_model.fit(dataset['train_data'], dataset['train_label'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['LogR_acc'] = LogR_model.score(dataset['test_data'], dataset['test_label'])

    ### KNN ###
    k_list = range(1, 101)
    knn_accuracy = []

    for k in k_list:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(dataset['train_data'], dataset['train_label'])

        prediction = knn_model.predict(dataset['test_data'])

        knn_accuracy.append(metrics.accuracy_score(prediction, dataset['test_label']))

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['maxKNN_acc'] = max(knn_accuracy)

    ### Gaussian Naive Bayes ###
    GaussiaNB_model = GaussianNB()
    GaussiaNB_model.fit(dataset['train_data'], dataset['train_label'])

    prediction = GaussiaNB_model.predict(dataset['test_data'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['GaussianNB_acc'] = metrics.accuracy_score(prediction, dataset['test_label'])

    ### Random Forest ###
    RandomForest = RandomForestClassifier(n_estimators=100)
    RandomForest.fit(dataset['train_data'], dataset['train_label'])

    prediction = RandomForest.predict(dataset['test_data'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['RandomForest_acc'] = metrics.accuracy_score(prediction, dataset['test_label'])

    ### Decision Tree ###
    DecisionTree = DecisionTreeClassifier()
    DecisionTree.fit(dataset['train_data'], dataset['train_label'])

    prediction = DecisionTree.predict(dataset['test_data'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['DecisionTree_acc'] = metrics.accuracy_score(prediction, dataset['test_label'])

    ### Support Vector Machine ###
    SVM = svm.SVC(kernel='rbf', C=1, gamma=0.1)
    SVM.fit(dataset['train_data'], dataset['train_label'])

    prediction = SVM.predict(dataset['test_data'])

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['SVM_acc'] = metrics.accuracy_score(prediction, dataset['test_label'])

# Plot various classification accuracy for each type of datasets
models = ['Logistic\nRegression', 'KNN', 'GaussianNB', 'Random\nForest', 'Decision\nTree', 'SVM']

width = 0.35
fg_ax4 = main_fig.add_subplot(gs[1, 0:6])
fg_ax4.bar(np.arange(len(models)) - width/3, [accuracy[0]['LogR_acc'], accuracy[0]['maxKNN_acc'], accuracy[0]['GaussianNB_acc'], accuracy[0]['RandomForest_acc'], accuracy[0]['DecisionTree_acc'], accuracy[0]['SVM_acc']], width=width/3, label='Original Dataset')
fg_ax4.bar(np.arange(len(models)), [accuracy[1]['LogR_acc'], accuracy[1]['maxKNN_acc'], accuracy[1]['GaussianNB_acc'], accuracy[1]['RandomForest_acc'], accuracy[1]['DecisionTree_acc'], accuracy[1]['SVM_acc']], width=width/3, label='Standardized Dataset')
fg_ax4.bar(np.arange(len(models)) + width/3, [accuracy[2]['LogR_acc'], accuracy[2]['maxKNN_acc'], accuracy[2]['GaussianNB_acc'], accuracy[2]['RandomForest_acc'], accuracy[2]['DecisionTree_acc'], accuracy[2]['SVM_acc']], width=width/3, label='Normalized Dataset')
fg_ax4.set_title('Classification Result Comparison : Original / Standardized / Normalized', fontdict={'weight':'bold', 'size' : 15})
plt.xticks(np.arange(len(models)), models, fontweight='bold')
plt.xlim(-0.3, 6.1)
plt.ylim(0, 1.1)
plt.ylabel('Accuracy', fontweight='bold')
fg_ax4.legend(loc='upper right')

plt.tight_layout()
plt.show()