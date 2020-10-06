
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
# 또한 Text 형태로 되어있는 성별 데이터는 숫자 형태로 전환 필요함

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
sex_mapping = {'male' : 0, 'female' : 1}				# 성별 데이터에 대한 숫자 값을 어떻게 전환할지 결정 (남성 0, 여성 1)
titanic_dataset['Sex'] = titanic_dataset['Sex'].map(sex_mapping)	# 데이터셋 내에서 성별에 해당되는 데이터를 숫자 형태로 Mapping하여 전환함

# Target 값인 승선원 생존 여부 데이터 Survived와 직접적으로 관련된 Pclass, Sex, Age 데이터만 Feature 데이터로 선정하여 데이터셋을 재구성함
# Target 값인 Survived은 Label 데이터셋으로 별도 분리함
print('[Pre-Processing] : Use Pclass, Sex, Age as dataset features, and Survived as Label / Filter out the rest')

# Pclass, Sex, Age로 구성된 Feature 데이터 구성
dataset_features = titanic_dataset.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Parch', 'Fare', 'SibSp'], axis=1)	

# Target 값인 Survived로 Label 데이터셋 구성
dataset_labels = titanic_dataset['Survived']	

print('[Pre-Processed Titanic Dataset]')
print(dataset_features)		# 전처리 후 데이터셋 출력하여 확인


### 데이터셋 Training / Test 분리 ###
# 데이터셋을 Training 70%와 Test 30%로 분리함
print('[Pre-Processing] : Split the dataset into 70% training dataset and 30% test dataset')
train_features, test_features, train_labels, test_labels = train_test_split(dataset_features, dataset_labels, test_size=0.3, random_state=32)

### KNN Classification을 통한 Prediction ###
# Titanic 데이터셋에 대해 K값을 바꿔가면서 KNN Classifcation 적용함
k_list = range(1, 101)
accuracy = []

# K의 값을 0에서 100까지 바꿔가면서 KNN Classification 수행함
for k in k_list:

    # 0에서 100 사이의 K값을 바꿔가면서 유방암 데이터셋에 대한 Fitting / Classification 수행
    classifier = KNeighborsClassifier(n_neighbors=k)

    # KNN Classifier에 대해 Fitting을 수행하여 K개 Neighbor 탐색에 용의한 자료 구조로 데이터를 배치함
    classifier.fit(train_features, train_labels)	

    # 0에서 100 사이의 K값을 바꿔가면서 KNN의 Classification 정확도를 저장함
    accuracy.append(classifier.score(test_features, test_labels))

# 최대 정확도를 가지는 K값을 찾아냄
for k in k_list:
    if accuracy[k-1] == max(accuracy):
        print('[Result] : K neighbor value with maximum accuracy = %d (%f)' % (k, accuracy[k-1]))
print('\n')

# K값이 변하면서 Classification 정확도가 어떻게 변하는지 확인하여 최대 정확도를 내는 K값을 선정할 수 있음
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

### 데이터셋 준비 및 전처리 ###
print('----- Iris Classification using KNN -----')

from sklearn.datasets import load_iris	# Iris Dataset을 scikit-learn 라이브러리에서 불러옴

iris_dataset = load_iris()		# Iris Dataset 준비

print(iris_dataset['DESCR'])		# Iris Dataset에 대한 설명 출력

print('Iris Dataset Key : ', iris_dataset.keys())			# Iris 데이터셋의 Attribute (구성요소) 출력

print('Iris Dataset Features : ', iris_dataset['feature_names'])	# Iris 데이터셋의 Feature 이름 출력
print('First 5 data in dataset \n', iris_dataset['data'][:5])		# Iris 데이터셋의 최초 5개 Feature 데이터 출력
print('Iris Dataset Target : ', iris_dataset['target_names'])		# Iris 데이터셋의 Target 이름 / Label 출력
print('First 5 targets in dataset \n', iris_dataset['target'][:5])	# Iris 데이터셋의 최초 5개 데이터의 Label 출력

### 데이터셋 Training / Test 분리 ###
# 데이터셋을 Training 70%와 Test 30%로 분리함
iris_train, iris_test, iris_label_train, iris_label_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.3, random_state=32)

### KNN Classification을 통한 Prediction ###
# Titanic 데이터셋에 대해 K값을 바꿔가면서 KNN Classifcation 적용함
k_list = range(1, 101)
accuracy = []

# K의 값을 0에서 100까지 바꿔가면서 KNN Classification 수행함
for k in k_list:

    # 0에서 100 사이의 K값을 바꿔가면서 유방암 데이터셋에 대한 Fitting / Classification 수행
    classifier = KNeighborsClassifier(n_neighbors=k)

    # KNN Classifier에 대해 Fitting을 수행하여 K개 Neighbor 탐색에 용의한 자료 구조로 데이터를 배치함
    classifier.fit(iris_train, iris_label_train)			

    # 0에서 100 사이의 K값을 바꿔가면서 KNN의 Classification 정확도를 저장함
    accuracy.append(classifier.score(iris_test, iris_label_test))

# 최대 정확도를 가지는 K값을 찾아냄
for k in k_list:
    if accuracy[k-1] == max(accuracy):
        print('[Result] : K neighbor value with maximum accuracy = %d (%f)' % (k, accuracy[k-1]))
print('\n')

# K값이 변하면서 Classification 정확도가 어떻게 변하는지 확인하여 최대 정확도를 내는 K값을 선정할 수 있음
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
