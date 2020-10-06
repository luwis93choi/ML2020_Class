########################################################################
### Q3. Breast Cancer Classification using Logistic Regression, KNN, ###
###     Gaussian Naive Bayes, Random Forest, Decision Tree, and SVM  ###
########################################################################

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

### 데이터셋 준비 및 전처리 ###
breast_cancer_dataset = load_breast_cancer()	# 유방암 데이터셋 준비

print('Breast Cancer Dataset Key : ', breast_cancer_dataset.keys())		# 유방암 데이터셋의 Attribute (구성요소) 출력

print('Breast Cancer Dataset Target : ', breast_cancer_dataset.target_names)	# 유방암 데이터셋의 Target/Label에 대한 내용 출력 (유방암 악성 여부)

breast_cancer_data = pd.DataFrame(breast_cancer_dataset.data)			# 유방암 데이터셋의 Feature 데이터를 Pandas DataFrame 형태로 전환
breast_cancer_label = pd.DataFrame(breast_cancer_dataset.target)		# 유방암 데이터셋의 Target/Label을 Pandas DataFrame 형태로 전환

print(breast_cancer_data.head())	# 유방암 데이터셋의 Feature 데이터 최초 5개를 출력함
					# 유방암 데이터셋의 30개의 Feature로 구성되어있다는 것을 알 수 있음

print(breast_cancer_label.head())	# 유방암 데이터셋의 Target/Label 데이터 최초 5개를 출력함

print(breast_cancer_data.describe())	# 유방암 데이터셋의 Feature 데이터에 대한 통계학적 정보를 출력함

### 데이터셋 Training/Test 분리 ###
# 데이터셋 데이터를 Training 70% : Test 30%로 분리함
train_data, test_data, train_label, test_label = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.3, random_state=23)

### 데이터셋 표준화 (Standardization) ###
# Feature 데이터에 대해서 Mean과 Standard Deviation를 이용한 z-score로 표준화를 수행함
# 각 Feature마다 다른 단위 체계를 사용하기 때문에 Scaling에 의해 데이터의 분포가 단위별로 다르게 나타나는 현상이 발생함
# Mean과 Standard Deviation을 통해 각 Feature에 해당되는 데이터 모음의 분포를 Mean 0, Standard Deviation 1으로 통일시킬 수 있음
# 각 Feature의 데이터 모음을 동일 분포로 통일시킴으로서 Scaling에 의한 부정적인 효과를 최소화함
scaler = StandardScaler()			# 데이터셋을 z-score로 전환할 수 있는 연산기 준비
train_data = scaler.fit_transform(train_data)	# Train 데이터에 대해 z-score로 표준화함
test_data = scaler.transform(test_data)		# Test 데이터에 대해 z-score로 표준화함

### 유방암 데이터셋에 대해 Logistic Regression 이용한 Classification 적용 ###
LogR_model = LogisticRegression()		# Logistic Regression Classifier 준비
LogR_model.fit(train_data, train_label)		# Train 데이터에 대해 Logistic Regression Fitting 수행

print('Logistic Regression Train Score : ', LogR_model.score(train_data, train_label))	# Train 데이터에 대한 Classification 정확도 출력
print('Logistic Regression Test Score : ', LogR_model.score(test_data, test_label))	# Test 데이터에 대한 Classification 정확도 출력
print('Logistic Regression Model Coefficient : ', LogR_model.coef_)			# 학습된 Logistic Regression 모델의 Coefficient 출력

LogR_acc = LogR_model.score(test_data, test_label)	# Logistic Regression 모델의 정확도 저장

### 유방암 데이터셋에 대해 KNN Classification 적용 ###
# K의 값을 0에서 100까지 바꿔가면서 KNN Classification 수행함
k_list = range(1, 101)
accuracy = []

for k in k_list:
    # 0에서 100 사이의 K값을 바꿔가면서 유방암 데이터셋에 대한 Fitting / Classification 수행
    knn_model = KNeighborsClassifier(n_neighbors=k)
    
    # KNN Classifier에 대해 Fitting을 수행하여 K개 Neighbor 탐색에 용의한 자료 구조로 데이터를 배치함
    knn_model.fit(train_data, train_label)

    # 0에서 100 사이의 K값을 바꿔가면서 Test 데이터에 대한 KNN의 Prediction 수행
    prediction = knn_model.predict(test_data)

    # 0에서 100 사이의 K값을 바꿔가면서 KNN의 Classification 정확도를 저장함
    accuracy.append(metrics.accuracy_score(prediction, test_label))

maxKNN_acc = max(accuracy)	# KNN이 가지는 최대 정확도 저장

### 유방암 데이터셋에 대해 Gaussian Naive Bayes 이용한 Classification 적용 ###
GaussiaNB_model = GaussianNB()				# Gaussian Naive Bayes Classifier 준비
GaussiaNB_model.fit(train_data, train_label)		# Train 데이터에 대해 Gaussian Naive Bayes Fitting 수행

prediction = GaussiaNB_model.predict(test_data)		# Test 데이터에 대해 Gaussian Naive Bayes Classifier의 Classification 수행

GaussianNB_acc = metrics.accuracy_score(prediction, test_label)	# Gaussian Naive Bayes 모델의 정확도 저장

### 유방암 데이터셋에 대해 Random Forest 이용한 Classification 적용 ###
RandomForest = RandomForestClassifier(n_estimators=100)		# Random Forest Classifier 준비
RandomForest.fit(train_data, train_label)			# Train 데이터에 대해 Random Forest Fitting 수행

prediction = RandomForest.predict(test_data)			# Test 데이터에 대해 Random Forest Classifier의 Classification 수행

RandomForest_acc = metrics.accuracy_score(prediction, test_label)	# Random Forest Classifier의 정확도 저장

### 유방암 데이터셋에 대해 Decision Tree 이용한 Classification 적용 ###
DecisionTree = DecisionTreeClassifier()			# Decision Tree Classifier 준비
DecisionTree.fit(train_data, train_label)		# Train 데이터에 대해 Decision Tree Fitting 수행

prediction = DecisionTree.predict(test_data)		# Test 데이터에 대해 Decision Tree Classifier의 Classification 수행

DecisionTree_acc = metrics.accuracy_score(prediction, test_label)	# Decision Tree Classifier의 정확도 저장

### 유방암 데이터셋에 대해 Support Vector Machine 이용한 Classification 적용 ###
SVM = svm.SVC(kernel='rbf', C=1, gamma=0.1)		# SVM Classifier 준비 / Non-linear Decision Boundary를 위한 kernel을 정의해줌
SVM.fit(train_data, train_label)			# Train 데이터에 대해 Label간 Margin이 최대가 될 수 있도록 Fitting 수행

prediction = SVM.predict(test_data)			# Test 데이터에 대해 SVM으로 Classification 수행

SVM_acc = metrics.accuracy_score(prediction, test_label)	# SVM Classifier의 정확도 저장



### 각각의 Classifier의 정확도를 막대 그래프로 표현하고, 각 정확도 값을 막대 그래프 위에 표시함 ###
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

# Logistic Regression Classifier의 정확도 막대 그래프
barlist[0].set_color('red')
plt.text(barlist[0].get_x(), accuracy[0] + 0.005, s=str(accuracy[0]), fontweight='bold')

# KNN Classifier의 정확도 막대 그래프
barlist[1].set_color('orange')
plt.text(barlist[1].get_x(), accuracy[1] + 0.005, s=str(accuracy[1]), fontweight='bold')

# Gaussian Naive Bayes Classifier의 정확도 막대 그래프
barlist[2].set_color('blue')
plt.text(barlist[2].get_x(), accuracy[2] + 0.005, s=str(accuracy[2]), fontweight='bold')

# Random Forest Classifier의 정확도 막대 그래프
barlist[3].set_color('green')
plt.text(barlist[3].get_x(), accuracy[3] + 0.005, s=str(accuracy[3]), fontweight='bold')

# Decision Tree Classifier의 정확도 막대 그래프
barlist[4].set_color('purple')
plt.text(barlist[4].get_x(), accuracy[4] + 0.005, s=str(accuracy[4]), fontweight='bold')

# SVM Classifier의 정확도 막대 그래프
barlist[5].set_color('yellow')
plt.text(barlist[5].get_x(), accuracy[5] + 0.005, s=str(accuracy[5]), fontweight='bold')

plt.xlabel('Classification Method')
plt.ylabel('Accuracy')
plt.title('Classification Results on Breast Cancer Dataset\n(Standardized Dataset)', fontsize='15', fontweight='bold')
plt.show()
