#####################################################################
### Breaset Cancer Classification using Logistic Regression, KNN, ###
### Gaussian Naive Bayes, Random Forest, Decision Tree, and SVM   ###
#####################################################################
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics

breaset_cancer_dataset = load_breast_cancer()

print('Breast Cancer Dataset Key : ', breaset_cancer_dataset.keys())

print('Breast Cancer Dataset Target : ', breaset_cancer_dataset.target_names)

breaset_cancer_data = pd.DataFrame(breaset_cancer_dataset.data)
breaset_cancer_label = pd.DataFrame(breaset_cancer_dataset.target)

print(breaset_cancer_data.head())

print(breaset_cancer_label.head())

print(breaset_cancer_data.describe())

train_data, test_data, train_label, test_label = train_test_split(breaset_cancer_data, breaset_cancer_label, test_size=0.3, random_state=23)

# Dataset Standardization 
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

### Logistic Regression ###
LogR_model = LogisticRegression()
LogR_model.fit(train_data, train_label)

print('Logistic Regression Train Score : ', LogR_model.score(train_data, train_label))
print('Logistic Regression Test Score : ', LogR_model.score(test_data, test_label))
print('Logistic Regression Model Coefficient : ', LogR_model.coef_)

LogR_acc = LogR_model.score(test_data, test_label)

### KNN ###
k_list = range(1, 101)
accuracy = []

for k in k_list:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_data, train_label)

    prediction = knn_model.predict(test_data)

    accuracy.append(metrics.accuracy_score(prediction, test_label))

maxKNN_acc = max(accuracy)

### Gaussian Naive Bayes ###
GaussiaNB_model = GaussianNB()
GaussiaNB_model.fit(train_data, train_label)

prediction = GaussiaNB_model.predict(test_data)

GaussianNB_acc = metrics.accuracy_score(prediction, test_label)

### Random Forest ###
RandomForest = RandomForestClassifier(n_estimators=100)
RandomForest.fit(train_data, train_label)

prediction = RandomForest.predict(test_data)

RandomForest_acc = metrics.accuracy_score(prediction, test_label)

### Decision Tree ###
DecisionTree = DecisionTreeClassifier()
DecisionTree.fit(train_data, train_label)

prediction = DecisionTree.predict(test_data)

DeicisionTree_acc = metrics.accuracy_score(prediction, test_label)

### Support Vector Machine ###
SVM = svm.SVC(kernel='rbf', C=1, gamma=0.1)
SVM.fit(train_data, train_label)

prediction = SVM.predict(test_data)

SVM_acc = metrics.accuracy_score(prediction, test_label)

print()
print('-----[Prediction Results on Breast Cancer Dataset]-----')
print('Prediction Accuracy of Logistic Accuracy : ', LogR_acc)
print('Maximum Prediction Accuracy of KNN : ', maxKNN_acc)
print('Prediction Accuracy of Gaussian NB : ', GaussianNB_acc)
print('Prediction Accuracy of Random Forest : ', RandomForest_acc)
print('Prediction Accuracy of Decision Tree : ', DeicisionTree_acc)
print('Prediction Accuracy of SVM : ', SVM_acc)
