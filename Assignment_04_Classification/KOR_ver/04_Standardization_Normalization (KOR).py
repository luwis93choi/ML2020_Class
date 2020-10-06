#################################################################################
### Q4. Breaset Cancer Classification using Standardization and Normalization ###
#################################################################################

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

### 데이터셋 준비 ###
breast_cancer_dataset = load_breast_cancer()	# 유방암 데이터셋 준비

print('Breast Cancer Dataset Key : ', breast_cancer_dataset.keys())		# 유방암 데이터셋의 Attribute (구성요소) 출력

print('Breast Cancer Dataset Target : ', breast_cancer_dataset.target_names)	# 유방암 데이터셋의 Target/Label에 대한 내용 출력 (유방암 악성 여부)

breast_cancer_data = pd.DataFrame(breast_cancer_dataset.data)			# 유방암 데이터셋의 Feature 데이터를 Pandas DataFrame 형태로 전환
breast_cancer_label = pd.DataFrame(breast_cancer_dataset.target)		# 유방암 데이터셋의 Target/Label을 Pandas DataFrame 형태로 전환

print(breast_cancer_data.head())	# 유방암 데이터셋의 Feature 데이터 최초 5개를 출력함
					# 유방암 데이터셋의 30개의 Feature로 구성되어있다는 것을 알 수 있음

print(breast_cancer_label.head())	# 유방암 데이터셋의 Target/Label 데이터 최초 5개를 출력함

print(breast_cancer_data.describe())	# 유방암 데이터셋의 Feature 데이터에 대한 통계학적 정보를 출력함

### 데이터셋 표준화 (Standardization) : Feature 데이터에 대해서 Mean과 Standard Deviation를 이용한 z-score로 표준화를 수행함 ###
mean = breast_cancer_dataset.data.mean(axis=0)			# Feature 데이터의 Mean을 산출함
standard_deviation = breast_cancer_dataset.data.std(axis=0)	# Feature 데이터의 Standard Deviation을 산출함

standardized_breast_cancer_data = (breast_cancer_dataset.data - mean) / standard_deviation	# (X-Mean) / Standard Deviation을 통해 z-score로 표준화함

### 데이터셋 정규화 Normalization : Feature 데이터에 대해서 Min-Max Normalization을 통해 모든 데이터를 0 ~ 1 사이 범위로 재정의함 ###
dataset_min = breast_cancer_dataset.data.min(axis=0)	# Feature 데이터의 최소값 산출
dataset_max = breast_cancer_dataset.data.max(axis=0)	# Feature 데이터의 최대값 산
normalized_breast_cancer_data = (breast_cancer_dataset.data - dataset_min) / (dataset_max - dataset_min)	# Min-Max Normalization으로 데이터셋 구성

# 기존 Feature 데이터, 표준화된 Feature 데이터, 정규화된 Feature 데이터를 비교하기 위해 각각에 대해 Box Plot 작성
main_fig = plt.figure(figsize=[20, 8])
gs = main_fig.add_gridspec(2, 6)	# Plotting 영역을 세부적으로 제어하기 위해 gridspec 사용함

# 기존 Feature 데이터에 대한 Box Plot 작성
fg_ax1 = main_fig.add_subplot(gs[0, 0:2])
fg_ax1.boxplot(breast_cancer_dataset.data)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax1.set_title('Original Breast Cancer Dataset\n(No Standardization or Normalization)', fontdict={'weight':'bold'})

# 표준화된 Feature 데이터에 대한 Box Plot 작성
fg_ax2 = main_fig.add_subplot(gs[0, 2:4])
fg_ax2.boxplot(standardized_breast_cancer_data)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax2.set_title('Standardized Breast Cancer Dataset', fontdict={'weight':'bold'})

# 정규화된 Feature 데이터에 대한 Box Plot 작성
fg_ax3 = main_fig.add_subplot(gs[0, 4:6])
fg_ax3.boxplot(normalized_breast_cancer_data)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax3.set_title('Normalized Breast Cancer Dataset', fontdict={'weight':'bold'})

### 데이터셋 Training / Test 분리 ###
# 기본 Feature 데이터, 표준화된 Feature 데이터, 정규환된 Feature 데이터를 각각 Training 70% : Test 30%로 분리함
# 기존 Feature 데이터를 Training 70% : Test 30%로 분리함
original_train_data, original_test_data, original_train_label, original_test_label = train_test_split(breast_cancer_data, 
												      breast_cancer_label, 
												      test_size=0.3, 
												      random_state=23)
# 표준화된 Feature 데이터를 Training 70% : Test 30%로 분리함
standardized_train_data, standardized_test_data, standardized_train_label, standardized_test_label= train_test_split(standardized_breast_cancer_data, 
														     breast_cancer_label, 
														     test_size=0.3, 
														     random_state=23)
# 정규화된 Feature 데이터를 Training 70% : Test 30%로 분리함
normalized_train_data, normalized_test_data, normalized_train_label, normalized_test_label = train_test_split(normalized_breast_cancer_data, 
													      breast_cancer_label, 
													      test_size=0.3, 
													      random_state=23)

# 각 데이터셋 종류별로 Train, Test를 정리하여 Dictionary 형태로 저장하고, 각각의 Classification에 적용함
dataset_group = [{'name' : 'original', 'train_data' : original_train_data, 'train_label' : original_train_label, 
		  'test_data' : original_test_data, 'test_label' : original_test_label},
                 {'name' : 'standardized', 'train_data' : standardized_train_data, 'train_label' : standardized_train_label, 
		  'test_data' : standardized_test_data, 'test_label' : standardized_test_label},
                 {'name' : 'normalized', 'train_data' : normalized_train_data, 'train_label' : normalized_train_label, 
		  'test_data' : normalized_test_data, 'test_label' : normalized_test_label}]

# 각 데이터셋 종류별로 적용한 Classification의 정확도를 각각 Dictionary 형태로 저장함
accuracy = [{'name' : 'original', 'LogR_acc' : None, 'maxKNN_acc' : None, 'GaussianNB_acc' : None, 
	     'RandomForest_acc' : None, 'DecisionTree_acc' : None, 'SVM_acc' : None},
            {'name' : 'standardized', 'LogR_acc' : None, 'maxKNN_acc' : None, 'GaussianNB_acc' : None, 
	     'RandomForest_acc' : None, 'DecisionTree_acc' : None, 'SVM_acc' : None},
            {'name' : 'normalized', 'LogR_acc' : None, 'maxKNN_acc' : None, 'GaussianNB_acc' : None, 
	     'RandomForest_acc' : None, 'DecisionTree_acc' : None, 'SVM_acc' : None}]

### 각 데이터셋 종류별로 Logistic Regression, KNN, Gaussian Naive Bayes, Random Forest, Decision Tree, SVM을 적용함 ###
for dataset in dataset_group:

    ### Logistic Regression 이용한 Classification 수행 ###
    LogR_model = LogisticRegression()					# Logistic Regression Classifier 준비
    LogR_model.fit(dataset['train_data'], dataset['train_label'])	# Train 데이터에 대해 Logistic Regression Fitting 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['LogR_acc'] = LogR_model.score(dataset['test_data'], dataset['test_label'])	# Logistic Regression 모델의 정확도 저장

    ### KNN 이용한 Classification 수행 ###
    # K의 값을 0에서 100까지 바꿔가면서 KNN Classification 수행함
    k_list = range(1, 101)
    knn_accuracy = []

    for k in k_list:
        # 0에서 100 사이의 K값을 바꿔가면서 유방암 데이터셋에 대한 Fitting / Classification 수행
        knn_model = KNeighborsClassifier(n_neighbors=k)

        # KNN Classifier에 대해 Fitting을 수행하여 K개 Neighbor 탐색에 용의한 자료 구조로 데이터를 배치함
        knn_model.fit(dataset['train_data'], dataset['train_label'])

        # 0에서 100 사이의 K값을 바꿔가면서 Test 데이터에 대한 KNN의 Prediction 수행
        prediction = knn_model.predict(dataset['test_data'])

        # 0에서 100 사이의 K값을 바꿔가면서 KNN의 Classification 정확도를 저장함
        knn_accuracy.append(metrics.accuracy_score(prediction, dataset['test_label']))

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['maxKNN_acc'] = max(knn_accuracy)	# KNN이 가지는 최대 정확도 저장

    ### Gaussian Naive Bayes 이용한 Classification 수행 ###
    GaussiaNB_model = GaussianNB()						# Gaussian Naive Bayes Classifier 준비
    GaussiaNB_model.fit(dataset['train_data'], dataset['train_label'])		# Train 데이터에 대해 Gaussian Naive Bayes Fitting 수행

    prediction = GaussiaNB_model.predict(dataset['test_data'])			# Test 데이터에 대해 Gaussian Naive Bayes Classifier의 Classification 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['GaussianNB_acc'] = metrics.accuracy_score(prediction, dataset['test_label'])	# Gaussian Naive Bayes 모델의 정확도 저장

    ### Random Forest 이용한 Classification 수행 ###
    RandomForest = RandomForestClassifier(n_estimators=100)			# Random Forest Classifier 준비
    RandomForest.fit(dataset['train_data'], dataset['train_label'])		# Train 데이터에 대해 Random Forest Fitting 수행

    prediction = RandomForest.predict(dataset['test_data'])			# Test 데이터에 대해 Random Forest Classifier의 Classification 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['RandomForest_acc'] = metrics.accuracy_score(prediction, dataset['test_label'])	# Random Forest Classifier의 정확도 저장

    ### Decision Tree 이용한 Classification 수행 ###
    DecisionTree = DecisionTreeClassifier()					# Decision Tree Classifier 준비
    DecisionTree.fit(dataset['train_data'], dataset['train_label'])		# Train 데이터에 대해 Decision Tree Fitting 수행

    prediction = DecisionTree.predict(dataset['test_data'])			# Test 데이터에 대해 Decision Tree Classifier의 Classification 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['DecisionTree_acc'] = metrics.accuracy_score(prediction, dataset['test_label'])	# Decision Tree Classifier의 정확도 저장

    ### Support Vector Machine 이용한 Classification 수행 ###
    SVM = svm.SVC(kernel='rbf', C=1, gamma=0.1)					# SVM Classifier 준비
    SVM.fit(dataset['train_data'], dataset['train_label'])			# Train 데이터에 대해 Label간 Margin이 최대가 될 수 있도록 Fitting 수행

    prediction = SVM.predict(dataset['test_data'])				# Test 데이터에 대해 SVM으로 Classification 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['SVM_acc'] = metrics.accuracy_score(prediction, dataset['test_label'])		# SVM Classifier의 정확도 저장

# 표준화와 정규화 적용 여부가 각각의 Classification 정확도에 어떤 영향을 주는지 비교하기 위해 각 데이터셋 별로 Classifier의 정확도를 Bar Plot으로 작성함
models = ['Logistic\nRegression', 'KNN', 'GaussianNB', 'Random\nForest', 'Decision\nTree', 'SVM']

width = 0.35
fg_ax4 = main_fig.add_subplot(gs[1, 0:6])
fg_ax4.bar(np.arange(len(models)) - width/3, [accuracy[0]['LogR_acc'], accuracy[0]['maxKNN_acc'], 
	   accuracy[0]['GaussianNB_acc'], accuracy[0]['RandomForest_acc'], accuracy[0]['DecisionTree_acc'], 
	   accuracy[0]['SVM_acc']], width=width/3, label='Original Dataset')

fg_ax4.bar(np.arange(len(models)), [accuracy[1]['LogR_acc'], accuracy[1]['maxKNN_acc'], 
	   accuracy[1]['GaussianNB_acc'], accuracy[1]['RandomForest_acc'], accuracy[1]['DecisionTree_acc'], 
	   accuracy[1]['SVM_acc']], width=width/3, label='Standardized Dataset')

fg_ax4.bar(np.arange(len(models)) + width/3, [accuracy[2]['LogR_acc'], accuracy[2]['maxKNN_acc'], 
	   accuracy[2]['GaussianNB_acc'], accuracy[2]['RandomForest_acc'], accuracy[2]['DecisionTree_acc'], 
	   accuracy[2]['SVM_acc']], width=width/3, label='Normalized Dataset')

fg_ax4.set_title('Classification Result Comparison : Original / Standardized / Normalized', fontdict={'weight':'bold', 'size' : 15})

plt.xticks(np.arange(len(models)), models, fontweight='bold')
plt.xlim(-0.3, 6.1)
plt.ylim(0, 1.1)
plt.ylabel('Accuracy', fontweight='bold')
fg_ax4.legend(loc='upper right')

plt.tight_layout()
plt.show()
