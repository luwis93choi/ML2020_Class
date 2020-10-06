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
from matplotlib import pyplot as plt

breast_cancer_dataset = load_breast_cancer()

print('Breast Cancer Dataset Key : ', breast_cancer_dataset.keys())

print('Breast Cancer Dataset Target : ', breast_cancer_dataset.target_names)

breast_cancer_data = pd.DataFrame(breast_cancer_dataset.data)
breast_cancer_label = pd.DataFrame(breast_cancer_dataset.target)

print(breast_cancer_data.head())

print(breast_cancer_label.head())

print(breast_cancer_data.describe())

train_data, test_data, train_label, test_label = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.3, random_state=23)

# Dataset Standardization 
scaler = StandardScaler()   # Standardize the dataset using mean and variance, and convert it into z-score values
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

DecisionTree_acc = metrics.accuracy_score(prediction, test_label)

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
print('Prediction Accuracy of Decision Tree : ', DecisionTree_acc)
print('Prediction Accuracy of SVM : ', SVM_acc)

models = ['Logistic\nRegression', 'KNN', 'GaussianNB', 'Random\nForest', 'Decision\nTree', 'SVM']
accuracy = [LogR_acc, maxKNN_acc, GaussianNB_acc, RandomForest_acc, DecisionTree_acc, SVM_acc]

plt.figure(figsize=[20, 8])

barlist = plt.bar(models, accuracy)

barlist[0].set_color('red')
plt.text(barlist[0].get_x(), accuracy[0] + 0.005, s=str(accuracy[0]), fontweight='bold')

barlist[1].set_color('orange')
plt.text(barlist[1].get_x(), accuracy[1] + 0.005, s=str(accuracy[1]), fontweight='bold')

barlist[2].set_color('blue')
plt.text(barlist[2].get_x(), accuracy[2] + 0.005, s=str(accuracy[2]), fontweight='bold')

barlist[3].set_color('green')
plt.text(barlist[3].get_x(), accuracy[3] + 0.005, s=str(accuracy[3]), fontweight='bold')

barlist[4].set_color('purple')
plt.text(barlist[4].get_x(), accuracy[4] + 0.005, s=str(accuracy[4]), fontweight='bold')

barlist[5].set_color('yellow')
plt.text(barlist[5].get_x(), accuracy[5] + 0.005, s=str(accuracy[5]), fontweight='bold')

plt.xlabel('Classification Method')
plt.ylabel('Accuracy')
plt.title('Classification Results on Breast Cancer Dataset\n(Standardized Dataset)', fontsize='15', fontweight='bold')
plt.show()