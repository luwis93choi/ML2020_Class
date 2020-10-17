####################################################################################
### Q3 Breast Cancer Classification using various Classifiers and Neural Network ###
####################################################################################

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

### 01 데이터셋 준비 단계 #################################################################################################################
breast_cancer_dataset = load_breast_cancer()	# 유방암 데이터셋 로딩

print('[Breast Cancer Classification using various Classifiers and Neural Network] \n')

# 유방암 데이터셋 Feature 이름 출력
print('Breast Cancer Dataset Features')
print(breast_cancer_dataset.feature_names, '\n')	

# 유방암 데이터셋 Label/Target 이름 출력
print('Breast Cancer Dataset Target : {} \n'.format(breast_cancer_dataset.target_names))

# 데이터셋 준비
breast_cancer_data = pd.DataFrame(breast_cancer_dataset.data)		# Feature 데이터를 pandas DataFrame 형태로 준비
breast_cancer_label = pd.DataFrame(breast_cancer_dataset.target)	# Target/Label 데이터를 pandas DataFrame 형태로 준비

# 데이터셋 표준화
stdScaler = StandardScaler()						# 데이터셋 표준화를 위한 Standard Scaler 준비
stdScaler.fit(breast_cancer_data)					# Feature 데이터의 Mean와 Standard Deviation 산출
breast_cancer_data_std = stdScaler.transform(breast_cancer_data)	# Feature 데이터의 Mean과 Standard Deviation을 적용하여 데이터셋 표준화

# Boxplot을 이용하여 비표준화 데이터셋과 표준화 데이터셋 비교
main_fig = plt.figure(figsize=[20, 8])
gs = main_fig.add_gridspec(2, 6)

# 비표준화 데이터셋 Boxplot 작성
fg_ax1 = main_fig.add_subplot(gs[0, 0:2])	
fg_ax1.boxplot(breast_cancer_dataset.data)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax1.set_title('Original Breast Cancer Dataset\n(No Standardization or Normalization)', fontdict={'weight':'bold'})

# 표준화 데이터셋 Boxplot 작성
fg_ax2 = main_fig.add_subplot(gs[0, 2:4])
fg_ax2.boxplot(breast_cancer_data_std)
plt.xticks(np.arange(len(breast_cancer_dataset.feature_names))+1, breast_cancer_dataset.feature_names, rotation=90)
fg_ax2.set_title('Standardized Breast Cancer Dataset', fontdict={'weight':'bold'})

'''
Boxplot을 통해 데이터셋이 표준화를 거치기 전에 일부 Feature에 대해 Scaling 범위 차이가 심하게 나타나는 것을 볼 수 있음
표준화를 통해 모든 Feature 데이터를 평균 0, 표준 편차 1인 z-score로 표준화를 시켜서 Scaling 문제를 해결함
'''

# Multi-Colinearity 확인을 위한 Correlation Matrix 작성
correlation_mat = np.corrcoef(breast_cancer_dataset.data.T)	# Feature 데이터에 대해 Coefficient Matrix 생성
fg_ax3 = main_fig.add_subplot(gs[0, 4:6])
fg_ax3.set_title('Breast Cancer - Correlation Coefficient', fontdict={'weight':'bold'})
plt.imshow(correlation_mat, interpolation='none', vmin=-1, vmax=1)	# Coefficient Matrix를 그래프로 출력
plt.colorbar(shrink=0.7)
plt.xticks(range(30),breast_cancer_dataset.feature_names,rotation=90,ha='center')
plt.yticks(range(30),breast_cancer_dataset.feature_names)

'''
Correlation Matrix 그래프를 통해 유방암 데이터셋의 일부 Feature들 간에 서로 Colinearity가 존재하는 것을 볼 수 있음
이러한 Multi-Colinearity 상황에서 Linear Regression 기반의 Classifier는 좋은 성능을 못 낼 것이라 예상됨
'''

# 비표준화 데이터셋, 표준화 데이터셋의 통합 관리를 위해 데이터셋 자료구조 생성
dataset_group = [{'name' : 'original', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None},
		 {'name' : 'standardized', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None}]

# 비표준화 데이터셋, 표준화 데이터셋을 각각 Training 70%, Test 30%로 분리함
# 비표준화 데이터셋 Training 70%, Test 30% 분리
(dataset_group[0]['X_train'], dataset_group[0]['X_test'], 
 dataset_group[0]['y_train'], dataset_group[0]['y_test']) = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.3, random_state=23)

# 표준화 데이터셋 Training 70%, Test 30% 분리
(dataset_group[1]['X_train'], dataset_group[1]['X_test'],
 dataset_group[1]['y_train'], dataset_group[1]['y_test']) = train_test_split(breast_cancer_data_std, breast_cancer_label, test_size=0.3, random_state=23)

# 각 데이터셋 종류와 Classifier 종류에 따른 정확도를 저장하기 위한 자료 구조 생성
accuracy = [{'name' : 'original', 'LinearR_acc' : None, 'LogR_acc' : None, 'maxKNN_acc' : None, 'GaussianNB_acc' : None, 'RandomForest_acc' : None, 
	     'DecisionTree_acc' : None, 'SVM_acc' : None, 'perceptron_acc' : None, 'MLP_acc' : None},
	    {'name' : 'standardized', 'LinearR_acc' : None, 'LogR_acc' : None, 'maxKNN_acc' : None, 'GaussianNB_acc' : None, 'RandomForest_acc' : None, 
	     'DecisionTree_acc' : None, 'SVM_acc' : None, 'perceptron_acc' : None, 'MLP_acc' : None}]

### 02 데이터셋 종류 및 Classifier 종류에 따른 학습 및 정확도 산출 단계 ###################################################################################

# 비표준화 데이터셋, 표준화 데이터셋 2가지 종류에 걸쳐서 각각의 Classifier를 학습 및 정확도 산출함
for dataset in dataset_group:

    ### Linear Regression 이용한 Classification ###
    LinearR_model = LinearRegression()
    LinearR_model.fit(dataset['X_train'], dataset['y_train'])	# Training 데이터셋을 이용한 Linear Regression 학습

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['LinearR_acc'] = LinearR_model.score(dataset['X_test'], dataset['y_test'])	# Test 데이터셋에 대한 Linear Regression 정확도 산출

    ### Logistic Regression 이용한 Classification ###
    LogR_model = LogisticRegression()
    LogR_model.fit(dataset['X_train'], dataset['y_train'])	# Training 데이터셋을 이용한 Logistic Regression 학습

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['LogR_acc'] = LogR_model.score(dataset['X_test'], dataset['y_test'])		# Test 데이터셋에 대한 Logistic Regression 정확도 산출

    ### KNN 이용한 Classification ###
    k_list = range(1, 101)	# KNN의 neighbor값을 1에서 100까지 바꿔가면서 최대 성능을 내는 neighbor 값을 산출할 예정
    knn_accuracy = []

    for k in k_list:
        knn_model = KNeighborsClassifier(n_neighbors=k)	
        knn_model.fit(dataset['X_train'], dataset['y_train'])	# neighbor값 k인 경우 Training 데이터셋을 이용하여 KNN 학습

        prediction = knn_model.predict(dataset['X_test'])	# Test 데이터셋에 대해 현재 학습된 KNN으로 Prediction 수행

        knn_accuracy.append(metrics.accuracy_score(prediction, dataset['y_test']))	# Test 데이터셋에 대한 현재 KNN의 정확도 산출

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['maxKNN_acc'] = max(knn_accuracy)	# 1에서 100 사이의 k값 중 최대 KNN 정확도를 산출하는 k값을 선정함

    ### Gaussian Naive Bayes 이용한 Classification ###
    GaussiaNB_model = GaussianNB()
    GaussiaNB_model.fit(dataset['X_train'], dataset['y_train'])		# Training 데이터셋을 이용한 Gaussian Naive Bayes 학습

    prediction = GaussiaNB_model.predict(dataset['X_test'])		# 학습된 Gaussian Naive Bayes의 Prediction 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['GaussianNB_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])	# 학습된 Gaussian Naive Bayes의 정확도 산출

    ### Random Forest 이용한 Classification ###
    RandomForest = RandomForestClassifier(n_estimators=100)
    RandomForest.fit(dataset['X_train'], dataset['y_train'])		# Training 데이터셋을 이용한 Random Forest 학습

    prediction = RandomForest.predict(dataset['X_test'])		# 학습된 Random Forest의 Prediction 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['RandomForest_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])	# 학습된 Random Forest의 정확도 산출

    ### Decision Tree 이용한 Classification ###
    DecisionTree = DecisionTreeClassifier()
    DecisionTree.fit(dataset['X_train'], dataset['y_train'])		# Training 데이터셋을 이용한 Decision Tree 학습

    prediction = DecisionTree.predict(dataset['X_test'])		# 학습된 Decision Tree의 Prediction 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['DecisionTree_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])	# 학습된 Decision Tree의 정확도 산출

    ### Support Vector Machine 이용한 Classification ###
    SVM = svm.SVC(kernel='rbf', C=1, gamma=0.1)			# Non-Linear한 데이터를 Linear하게 Refine하기 위한 kernel을 반영하여 SVM 생성
    SVM.fit(dataset['X_train'], dataset['y_train'])		# Training 데이터셋을 이용한 SVM 학습

    prediction = SVM.predict(dataset['X_test'])			# 학습된 SVM의 Prediction 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['SVM_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])	# 학습된 SVM의 정확도 산출

    ### Perceptron 이용한 Classification ###
     # 단일 Perceptron 선언 (최대 반복 횟수 40, Learning Rate 0.001, 목표 Loss 0.001, Random State (Seed) 23)
    perceptron = Perceptron(max_iter=40, eta0=0.001, tol=1e-3, random_state=23)
	
    perceptron.fit(dataset['X_train'], dataset['y_train'])	# Training 데이터셋을 이용한 Perceptron 학습

    prediction = perceptron.predict(dataset['X_test'])		# 학습된 Perceptron의 Prediction 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['perceptron_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])	# 학습된 Perceptron의 정확도 산출

    ### Multi-Layer Perceptron 이용한 Classification ###
     # Multi-Layer Perceptron 선언 (Hidden Layer 2개, 각각 Neuron 100개씩 사용, Random State (Seed) 23)
    mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=23)

    mlp.fit(dataset['X_train'], dataset['y_train'])	# Training 데이터셋을 이용한 Multi-Layer Perceptron 학습

    prediction = mlp.predict(dataset['X_test'])		# 학습된 Multi-Layer Perceptron의 Prediction 수행

    for acc in accuracy:
        if acc['name'] == dataset['name']:
            acc['MLP_acc'] = metrics.accuracy_score(prediction, dataset['y_test'])	# 학습된 Multi-Layer Perceptron의 정확도 산출

### 03 데이터셋 종류 및 Classifier 종류에 따른 정확도 그래프 출력을 통한 비교 ######################################################################################

# 사용한 Classifier 종류
models = ['Linear\nRegression', 'Logistic\nRegression', 'KNN', 'GaussianNB', 'Random\nForest', 'Decision\nTree', 'SVM', 'Perceptron', 'MLP\nLayer : [100, 100]']

width = 0.35
fg_ax4 = main_fig.add_subplot(gs[1, 0:6])

# 비표준화 데이터셋을 사용했을 경우 각 Classifier별 정확도 그래프 출력
fg_ax4.bar(np.arange(len(models)) - width/4, [accuracy[0]['LinearR_acc'], accuracy[0]['LogR_acc'], accuracy[0]['maxKNN_acc'], 
					      accuracy[0]['GaussianNB_acc'], accuracy[0]['RandomForest_acc'], accuracy[0]['DecisionTree_acc'], 
					      accuracy[0]['SVM_acc'], accuracy[0]['perceptron_acc'], accuracy[0]['MLP_acc']], 
					      width=width/3, label='Original Dataset')

# 표준화 데이터셋을 사용했을 경우 각 Classifier별 정확도 그래프 출력
fg_ax4.bar(np.arange(len(models)) + width/4, [accuracy[1]['LinearR_acc'], accuracy[1]['LogR_acc'], accuracy[1]['maxKNN_acc'], 
					      accuracy[1]['GaussianNB_acc'], accuracy[1]['RandomForest_acc'], accuracy[1]['DecisionTree_acc'], 
					      accuracy[1]['SVM_acc'], accuracy[1]['perceptron_acc'], accuracy[1]['MLP_acc']], 
					      width=width/3, label='Standardized Dataset')

fg_ax4.set_title('Classification Result Comparison : Original / Standardized', fontdict={'weight':'bold', 'size' : 15})
plt.xticks(np.arange(len(models)), models, fontweight='bold')
plt.xlim(-0.3, 10.1)
plt.ylim(0, 1.1)
plt.ylabel('Accuracy', fontweight='bold')
fg_ax4.legend(loc='upper right')

# 비표준화/표준화 데이터셋에 대한 각 Classifier별 정확도값을 Textbox로 출력
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





'''
그래프 확인 결과 유방암 데이터셋에 대해 표준화 여부에 상관없이 Linear Regression은 0.7 정도의 낮은 정확도를 보이고 있음.
이러한 이유는 데이터셋의 Feature 간에 Multi-Colinearity가 존재하기에 Linear한 Classification이 힘들기 때문임.
또한 표준화는 Feature 데이터에 대해 Scaling을 통일시켜주지만 Multi-Colinearity를 해결하지 못함

비표준화 데이터에 대해서도 Non-Linear한 데이터에 대한 Classification 능력을 갖추고 있는 Logistics Regression, KNN, Gaussain Naive Bayes, Random Forest,
Decision Tree, Perceptron, Multi-Layer Perceptron이 0.9 정도의 높은 정확도를 보여줄 수 있음.

그러나 비표준화 상황에서 SVM은 매우 낮은 정확도를 보이고 있음. 왜냐하면 SVM은 Label/Target Class간에 Margin이 최대가 되는 방향으로 Classification을 하는 상황에서
Feature의 Scale 차이로 인해 Fitting이 제대로 되지 않기 때문임.

데이터셋에 대한 표준화를 수행하게되면 SVM을 포함한 대부분의 Classifier 성능이 개선되거나 0.9 정도의 높은 정확도를 유지하는 것을 볼 수 있음.
특히 SVM의 경우 표준화를 통한 Rescaling을 통해 정확도가 크게 개선되는 것을 볼 수 있음.
'''
