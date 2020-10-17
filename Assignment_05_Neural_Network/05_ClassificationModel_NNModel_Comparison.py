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

fg_ax1 = main_fig.add_subplot(gs[0, 0:3])
fg_ax1.boxplot(breast_cancer_dataset.data)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax1.set_title('Original Breast Cancer Dataset\n(No Standardization or Normalization)', fontdict={'weight':'bold'})

fg_ax2 = main_fig.add_subplot(gs[0, 3:6])
fg_ax2.boxplot(breast_cancer_data_std)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax2.set_title('Standardized Breast Cancer Dataset', fontdict={'weight':'bold'})

# Dataset Group 
dataset_group = [{'name' : 'original', 'X_train' : 1, 'y_train' : 2, 'X_test' : 3, 'y_test' : 4},
                 {'name' : 'starndardized', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None}]

# Split original / standardized breast cancer dataset into 70% training set and 30% test set
(dataset_group['name' == 'original']['X_train'], dataset_group['name' == 'original']['X_test'], 
 dataset_group['name' == 'original']['y_train'], dataset_group['name' == 'original']['y_test']) = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.3, random_state=23)

(dataset_group['name' == 'starndardized']['X_train'], dataset_group['name' == 'starndardized']['X_test'], 
 dataset_group['name' == 'starndardized']['y_train'], dataset_group['name' == 'starndardized']['y_test']) = train_test_split(breast_cancer_data_std, breast_cancer_label, test_size=0.3, random_state=23)

### Classifier training & Prediction over all dataset groups ###


plt.show()