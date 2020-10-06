
####################################################
### Q2. Titanic Classification Example using KNN ###
####################################################
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

### 데이터셋 준비 및 전처리 ###
# Titanic Dataset이 저장된 csv 파일을 읽어옴
print('----- Titanic Classification using KNN -----')
titanic_dataset = pd.read_csv('./train.csv')

# Titanic Dataset 출력
print(titanic_dataset)

# Titanic Dataset 형태 출력
print(titanic_dataset.shape)

### 전처리 필요 여부 확인 : 데이터셋 내 NULL 및 NaN 값 존재 여부 확인 ###
print('[Pre-Processing] : Before pre-processing NaN values of Age column')
print(titanic_dataset.isnull().sum())
# 확인 결과 Age에 대해 177개, Cabin에 대해 687개, Embarked에 대해 2개 데이터가 NULL 또는 NaN으로 전처리 필요함

### Age 데이터에 대한 전처리 수행 ###
# 데이터셋 내에서 숫자로 존재해야하는 Age 데이터가 NULL 또는 NaN 같이 존재하지 않는 경우에는 해당 데이터의 성별의 평균 나이값으로 대체함
print('[Pre-Processing] : Replace NaN values of Age column as average age of gender')
titanic_dataset['Age'].fillna(titanic_dataset.groupby('Sex')['Age'].transform('mean'), inplace=True)
print(titanic_dataset.isnull().sum())	# 전처리 필요 여부 재확인

### Cabin, Embarked에 대한 전처리 수행 ###
print('[Pre-Processing] : Drop Cabin and Embarked Features')
titanic_dataset = titanic_dataset.drop(['Cabin', 'Embarked'], axis=1)	# Cabin, Embarked는 Age와 다르게 대체할 수 있는 방법이 없기에 데이터셋에서 제외함
print(titanic_dataset.isnull().sum())	# 전처리 필요 여부 재확인

### Text로 구성된 Feature에 대해 숫자 형태로 전환 ###
print('[Pre-Processing] : Convert text-based features into numeric labels')
sex_mapping = {'male' : 0, 'female' : 1}
titanic_dataset['Sex'] = titanic_dataset['Sex'].map(sex_mapping)

print('[Pre-Processing] : Use Pclass, Sex, Age as dataset features, and Survived as Label / Filter out the rest')
dataset_features = titanic_dataset.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Parch', 'Fare', 'SibSp'], axis=1)
dataset_labels = titanic_dataset['Survived']

print('[Pre-Processed Titanic Dataset]')
print(dataset_features)

print('[Pre-Processing] : Split the dataset into 70% training dataset and 30% validation dataset')
train_features, test_features, train_labels, test_labels = train_test_split(dataset_features, dataset_labels, test_size=0.3, random_state=32)

k_list = range(1, 101)
accuracy = []

for k in k_list:

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_features, train_labels)
    accuracy.append(classifier.score(test_features, test_labels))

for k in k_list:
    if accuracy[k-1] == max(accuracy):
        print('[Result] : K neighbor value with maximum accuracy = %d (%f)' % (k, accuracy[k-1]))
print('\n')

# Plot the changes in accuracy according to the changes in the number of neighbors
plt.figure(figsize=[20, 8])

plt.subplot(1, 2, 1)
plt.plot(k_list, accuracy, 'bo-')
plt.xlabel('Number of Neigbhors in KNN')
plt.ylabel('Validation Accuracy')
plt.title('Titanic KNN Classifier Accuracy', fontsize=15, fontweight='bold')
plt.text(5, 0.05, 'Features : Pclass, Sex, Age \nLabel : Survived (0 or 1)', bbox=dict(facecolor='white', alpha=0.5), fontsize=13)
plt.xlim(1, 100)
plt.ylim(0.0, 1.1)
plt.grid()

#################################################
### Q2. Iris Classification Example using KNN ###
#################################################

print('----- Iris Classification using KNN -----')

from sklearn.datasets import load_iris

iris_dataset = load_iris()

print(iris_dataset['DESCR'])

print('Iris Dataset Key : ', iris_dataset.keys())

print('Iris Dataset Features : ', iris_dataset['feature_names'])
print('First 5 data in dataset \n', iris_dataset['data'][:5])
print('Iris Dataset Target : ', iris_dataset['target_names'])
print('First 5 targets in dataset \n', iris_dataset['target'][:5])

iris_train, iris_test, iris_label_train, iris_label_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.3, random_state=32)

k_list = range(1, 101)
accuracy = []

for k in k_list:

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(iris_train, iris_label_train)
    accuracy.append(classifier.score(iris_test, iris_label_test))

for k in k_list:
    if accuracy[k-1] == max(accuracy):
        print('[Result] : K neighbor value with maximum accuracy = %d (%f)' % (k, accuracy[k-1]))
print('\n')

# Plot the changes in accuracy according to the changes in the number of neighbors
sub2 = plt.subplot(1, 2, 2)
plt.plot(k_list, accuracy, 'bo-')
plt.xlabel('Number of Neigbhors in KNN')
plt.ylabel('Validation Accuracy')
plt.title('Iris KNN Classifier Accuracy', fontsize=15, fontweight='bold')
plt.text(5, 0.05, 'Features : sepal length, sepal width, petal length, petal width) \nLabel : setosa (0), versicolor (1), virginica (2)', bbox=dict(facecolor='white', alpha=0.5), fontsize=13)
plt.xlim(1, 100)
plt.ylim(0.0, 1.1)
plt.grid()

plt.show()
