import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

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

#######################################################################################################################################
### 01 데이터셋 준비 단계 #################################################################################################################
#######################################################################################################################################
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

# 비표준화 데이터셋, 표준화 데이터셋의 통합 관리를 위해 데이터셋 자료구조 생성
dataset_group = [{'name' : 'Original', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None},
		         {'name' : 'Standardized', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None}]

# 비표준화 데이터셋, 표준화 데이터셋을 각각 Training 70%, Test 30%로 분리함
# 비표준화 데이터셋 Training/Validation 70%, Test 30% 분리
(dataset_group[0]['X_train'], dataset_group[0]['X_test'], 
 dataset_group[0]['y_train'], dataset_group[0]['y_test']) = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.3, random_state=23)

# 표준화 데이터셋 Training/Validation 70%, Test 30% 분리
(dataset_group[1]['X_train'], dataset_group[1]['X_test'],
 dataset_group[1]['y_train'], dataset_group[1]['y_test']) = train_test_split(breast_cancer_data_std, breast_cancer_label, test_size=0.3, random_state=23)

# 각 데이터셋 종류와 Classifier 종류에 따른 정확도를 저장하기 위한 자료 구조 생성
accuracy = [{'name' : 'Original', 'LinearR_acc' : None, 'LogR_acc' : None, 'KNN_acc' : None, 'GaussianNB_acc' : None, 'RandomForest_acc' : None, 
	                              'SVM_acc' : None, 'MLP_acc' : None},
	        {'name' : 'Standardized', 'LinearR_acc' : None, 'LogR_acc' : None, 'KNN_acc' : None, 'GaussianNB_acc' : None, 'RandomForest_acc' : None, 
	                                  'SVM_acc' : None, 'MLP_acc' : None}]

predictions = [{'name' : 'Original', 'LinearR_predict' : None, 'LogR_predict' : None, 'KNN_predict' : None, 'GaussianNB_predict' : None, 'RandomForest_predict' : None, 
	                                 'SVM_predict' : None, 'MLP_predict' : None},
	           {'name' : 'Standardized', 'LinearR_predict' : None, 'LogR_predict' : None, 'KNN_predict' : None, 'GaussianNB_predict' : None, 'RandomForest_predict' : None, 
	                                     'SVM_predict' : None, 'MLP_predict' : None}]

#######################################################################################################################################
### 02 Dataset Analysis ###############################################################################################################
#######################################################################################################################################

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

plt.tight_layout()
plt.show()

################################################################################################################
### 03 Principle Component Analysis & Top 2 PC Visualization ###################################################
################################################################################################################

plt.cla()
classifier_num = 0

pca_original = PCA(n_components=10, random_state=42).fit(dataset_group[0]['X_train'])
pca_std = PCA(n_components=10, random_state=42).fit(dataset_group[1]['X_train'])

plt.plot(range(1, len(pca_original.explained_variance_ratio_)+1), pca_original.explained_variance_ratio_, '*-', 
               label='PC on Original Dataset')
plt.plot(range(1, len(pca_std.explained_variance_ratio_)+1), pca_std.explained_variance_ratio_, '*-', 
               label='PC on Standardized Dataset')
plt.xlabel('Principle Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(pca_original.explained_variance_ratio_)+1))
plt.title('Explained Variance Ratio of PC')
plt.legend(loc="upper right")
plt.show()


# 비표준화 데이터셋, 표준화 데이터셋의 통합 관리를 위해 데이터셋 자료구조 생성
PCA_dataset_group = [{'name' : 'Original', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None},
		             {'name' : 'Standardized', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None}]

pca_original = PCA(n_components=2, random_state=42).fit(dataset_group[0]['X_train'])
pca_std = PCA(n_components=5, random_state=42).fit(dataset_group[1]['X_train'])

PCA_dataset_group[0]['X_train'] = pca_original.transform(dataset_group[0]['X_train'])
PCA_dataset_group[0]['y_train'] = dataset_group[0]['y_train']
PCA_dataset_group[1]['X_train'] = pca_original.transform(dataset_group[1]['X_train'])
PCA_dataset_group[1]['y_train'] = dataset_group[1]['y_train']

PCA_dataset_group[0]['X_test'] = pca_original.transform(dataset_group[0]['X_test'])
PCA_dataset_group[0]['y_test'] = dataset_group[0]['y_test']
PCA_dataset_group[1]['X_test'] = pca_original.transform(dataset_group[1]['X_test'])
PCA_dataset_group[1]['y_test'] = dataset_group[1]['y_test']

PCA_breast_cancer_data = pca_original.transform(breast_cancer_data)
PCA_breast_cancer_data_std = pca_original.transform(breast_cancer_data_std)

#######################################################################################################################################
### 04 데이터셋 종류 및 Classifier 종류에 따른 학습 및 정확도 산출 단계 #########################################################################
#######################################################################################################################################

plt.cla()
classifier_num = 0

# 비표준화 데이터셋, 표준화 데이터셋 2가지 종류에 걸쳐서 각각의 Classifier를 학습 및 정확도 산출함
for dataset in dataset_group:

    print('{}'.format(dataset['name']))

    ### Linear Regression 이용한 Classification ###
    print('------ Linear Regression ------')
    LinearR_model = LinearRegression()
    params = {}
    GridSearch_CrossValidation = GridSearchCV(LinearR_model, params, scoring='r2', verbose=0)
    GridSearch_CrossValidation.fit(dataset['X_train'], np.ravel(dataset['y_train'], order='C'))
    
    print('Best Training Accuracy : {}'.format(GridSearch_CrossValidation.best_score_))
    LinearR_final = GridSearch_CrossValidation.best_estimator_

    LinearR_model_prediction = LinearR_final.predict(dataset['X_test'])	

    print('Test Accuracy : {}'.format(metrics.r2_score(LinearR_model_prediction, dataset['y_test'])))	

    accuracy['name'==dataset['name']]['Linear_acc'] = metrics.r2_score(LinearR_model_prediction, dataset['y_test'])
    predictions['name'==dataset['name']]['Linear_predict'] = LinearR_model_prediction

    ### Logistic Regression 이용한 Classification ###
    print('------ Logistics Regression ------')
    LogR_model = LogisticRegression()
    params = {}
    GridSearch_CrossValidation = GridSearchCV(LogR_model, params, scoring='accuracy', verbose=0)
    GridSearch_CrossValidation.fit(dataset['X_train'], np.ravel(dataset['y_train'], order='C'))

    print('Optimal Params : {}'.format(GridSearch_CrossValidation.best_params_))
    print('Best Training Accuracy : {}'.format(GridSearch_CrossValidation.best_score_))
    LogR_final = GridSearch_CrossValidation.best_estimator_

    LogR_model_prediction = LogR_final.predict(dataset['X_test'])	

    print('Test Accuracy : {}'.format(metrics.accuracy_score(LogR_model_prediction, dataset['y_test'])))	

    accuracy['name'==dataset['name']]['LogR_acc'] = metrics.accuracy_score(LogR_model_prediction, dataset['y_test'])
    predictions['name'==dataset['name']]['LogR_predict'] = LogR_model_prediction

    confusion_Mat = confusion_matrix(dataset['y_test'], LogR_model_prediction)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ["{0:.2%}".format(value) for value in confusion_Mat.flatten()/np.sum(confusion_Mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_Mat/np.sum(confusion_Mat), annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix of Best Logistics Regression Model\n[' + dataset['name'] + ' Dataset + ' + 'Grid Search + Cross Validation]')
    plt.show()

    classifier_num += 1
    
    ### KNN : Grid Search + Cross Validation Training / Evaluation with Test Set ###
    print('------ KNN ------')
    knn_model = KNeighborsClassifier()
    params = {'n_neighbors':range(1, 101)}
    GridSearch_CrossValidation = GridSearchCV(knn_model, params, scoring='accuracy', verbose=0)
    GridSearch_CrossValidation.fit(dataset['X_train'], np.ravel(dataset['y_train'], order='C'))

    print('Optimal Params : {}'.format(GridSearch_CrossValidation.best_params_))
    print('Best Training Accuracy : {}'.format(GridSearch_CrossValidation.best_score_))
    knn_final = GridSearch_CrossValidation.best_estimator_

    knn_model_prediction = knn_final.predict(dataset['X_test'])	

    print('Test Accuracy : {}'.format(metrics.accuracy_score(knn_model_prediction, dataset['y_test'])))	

    accuracy['name'==dataset['name']]['KNN_acc'] = metrics.accuracy_score(knn_model_prediction, dataset['y_test'])
    predictions['name'==dataset['name']]['KNN_predict'] = knn_model_prediction

    confusion_Mat = confusion_matrix(dataset['y_test'], knn_model_prediction)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ["{0:.2%}".format(value) for value in confusion_Mat.flatten()/np.sum(confusion_Mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_Mat/np.sum(confusion_Mat), annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix of Best KNN Model\n[' + dataset['name'] + ' Dataset + ' + 'Grid Search + Cross Validation]')
    plt.show()

    classifier_num += 1

    ### Gaussian Naive Bayes : Grid Search + Cross Validation Training / Evaluation with Test Set ###
    print('------ Gaussian Naive Bayes ------')
    GaussiaNB_model = GaussianNB()
    params = {}
    GridSearch_CrossValidation = GridSearchCV(GaussiaNB_model, params, scoring='accuracy', verbose=0)
    GridSearch_CrossValidation.fit(dataset['X_train'], np.ravel(dataset['y_train'], order='C'))
    
    print('Optimal Params : {}'.format(GridSearch_CrossValidation.best_params_))
    print('Best Training Accuracy : {}'.format(GridSearch_CrossValidation.best_score_))
    GaussiaNB_final = GridSearch_CrossValidation.best_estimator_

    GaussiaNB_model_prediction = GaussiaNB_final.predict(dataset['X_test'])	

    print('Test Accuracy : {}'.format(metrics.accuracy_score(GaussiaNB_model_prediction, dataset['y_test'])))	

    accuracy['name'==dataset['name']]['GaussianNB_acc'] = metrics.accuracy_score(GaussiaNB_model_prediction, dataset['y_test'])
    predictions['name'==dataset['name']]['GaussianNB_predict'] = GaussiaNB_model_prediction

    confusion_Mat = confusion_matrix(dataset['y_test'], GaussiaNB_model_prediction)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ["{0:.2%}".format(value) for value in confusion_Mat.flatten()/np.sum(confusion_Mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_Mat/np.sum(confusion_Mat), annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix of Best Gaussian Naive Bayes Model\n[' + dataset['name'] + ' Dataset + ' + 'Grid Search + Cross Validation]')
    plt.show()

    classifier_num += 1

    ### Random Forest : Grid Search + Cross Validation Training / Evaluation with Test Set ###
    print('------ Random Forest ------')
    RandomForest = RandomForestClassifier()
    params = {'n_estimators' : range(1, 101)}
    GridSearch_CrossValidation = GridSearchCV(RandomForest, params, scoring='accuracy', verbose=0)
    GridSearch_CrossValidation.fit(dataset['X_train'], np.ravel(dataset['y_train'], order='C'))
    
    print('Optimal Params : {}'.format(GridSearch_CrossValidation.best_params_))
    print('Best Training Accuracy : {}'.format(GridSearch_CrossValidation.best_score_))
    RandomForest_final = GridSearch_CrossValidation.best_estimator_

    RandomForest_prediction = RandomForest_final.predict(dataset['X_test'])	

    print('Test Accuracy : {}'.format(metrics.accuracy_score(RandomForest_prediction, dataset['y_test'])))	

    accuracy['name'==dataset['name']]['RandomForest_acc'] = metrics.accuracy_score(RandomForest_prediction, dataset['y_test'])
    predictions['name'==dataset['name']]['RandomForest_predict'] = RandomForest_prediction

    confusion_Mat = confusion_matrix(dataset['y_test'], RandomForest_prediction)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ["{0:.2%}".format(value) for value in confusion_Mat.flatten()/np.sum(confusion_Mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_Mat/np.sum(confusion_Mat), annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix of Best Random Forest Model\n[' + dataset['name'] + ' Dataset + ' + 'Grid Search + Cross Validation]')
    plt.show()

    classifier_num += 1

    ### Support Vector Machine : Grid Search + Cross Validation Training / Evaluation with Test Set ###
    print('------ SVM ------')
    SVM = svm.SVC()
    params = {'kernel' : ['linear', 'rbf', 'sigmoid'], 'C' : [0.001, 0.01, 0.1, 1.0], 'gamma' : [0.001, 0.01, 0.1, 1.0]}
    GridSearch_CrossValidation = GridSearchCV(SVM, params, scoring='accuracy', verbose=0)
    GridSearch_CrossValidation.fit(dataset['X_train'], np.ravel(dataset['y_train'], order='C'))
    
    print('Optimal Params : {}'.format(GridSearch_CrossValidation.best_params_))
    print('Best Training Accuracy : {}'.format(GridSearch_CrossValidation.best_score_))
    SVM_final = GridSearch_CrossValidation.best_estimator_

    SVM_prediction = SVM_final.predict(dataset['X_test'])	

    print('Test Accuracy : {}'.format(metrics.accuracy_score(SVM_prediction, dataset['y_test'])))	

    accuracy['name'==dataset['name']]['SVM_acc'] = metrics.accuracy_score(SVM_prediction, dataset['y_test'])
    predictions['name'==dataset['name']]['SVM_predict'] = SVM_prediction

    confusion_Mat = confusion_matrix(dataset['y_test'], SVM_prediction)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ["{0:.2%}".format(value) for value in confusion_Mat.flatten()/np.sum(confusion_Mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_Mat/np.sum(confusion_Mat), annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix of Best SVM Model\n[' + dataset['name'] + ' Dataset + ' + 'Grid Search + Cross Validation]')
    plt.show()

    classifier_num += 1

    ### Multi-Layer Perceptron : Grid Search + Cross Validation Training / Evaluation with Test Set ###
    print('------ MLP ------')
    MLP = MLPClassifier()
    params = {'hidden_layer_sizes' : [[1, 10], [1, 100], [10, 100], [100, 100], [100, 10], [100, 1], [10, 1]], 
              'solver' : ['lbfgs', 'sgd', 'adam'],
              'random_state' : [23]}
    GridSearch_CrossValidation = GridSearchCV(MLP, params, scoring='accuracy', verbose=0)
    GridSearch_CrossValidation.fit(dataset['X_train'], np.ravel(dataset['y_train'], order='C'))
    
    print('Optimal Params : {}'.format(GridSearch_CrossValidation.best_params_))
    print('Best Training Accuracy : {}'.format(GridSearch_CrossValidation.best_score_))
    MLP_final = GridSearch_CrossValidation.best_estimator_

    MLP_prediction = MLP_final.predict(dataset['X_test'])	

    print('Test Accuracy : {}'.format(metrics.accuracy_score(MLP_prediction, dataset['y_test'])))	

    accuracy['name'==dataset['name']]['MLP_acc'] = metrics.accuracy_score(MLP_prediction, dataset['y_test'])
    predictions['name'==dataset['name']]['MLP_predict'] = MLP_prediction

    confusion_Mat = confusion_matrix(dataset['y_test'], MLP_prediction)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ["{0:.2%}".format(value) for value in confusion_Mat.flatten()/np.sum(confusion_Mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_Mat/np.sum(confusion_Mat), annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix of Best MLP Model\n[' + dataset['name'] + ' Dataset + ' + 'Grid Search + Cross Validation]')
    plt.show()

    classifier_num += 1

cmap = plt.cm.rainbow
for prediction, dataset in predictions, dataset_group:

    fpr = [0 for i in range(int(classifier_num))]
    tpr = [0 for i in range(int(classifier_num))]
    roc_auc = [0 for i in range(int(classifier_num))]

    i = 0

    fpr[i], tpr[i], _ = roc_curve(prediction['LogR_predict'], dataset['y_test'])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, c=cmap(i/classifier_num), label='Logistic Regression ROC curve (area = %0.2f)' % roc_auc[i])
    i += 1

    fpr[i], tpr[i], _ = roc_curve(prediction['KNN_predict'], dataset['y_test'])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, c=cmap(i/classifier_num), label='KNN ROC curve (area = %0.2f)' % roc_auc[i])
    i += 1

    fpr[i], tpr[i], _ = roc_curve(prediction['GaussianNB_predict'], dataset['y_test'])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, c=cmap(i/classifier_num), label='Gaussian Naive Bayes ROC curve (area = %0.2f)' % roc_auc[i])
    i += 1

    fpr[i], tpr[i], _ = roc_curve(prediction['RandomForest_predict'], dataset['y_test'])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, c=cmap(i/classifier_num), label='Random Forest ROC curve (area = %0.2f)' % roc_auc[i])
    i += 1

    fpr[i], tpr[i], _ = roc_curve(prediction['SVM_predict'], dataset['y_test'])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, c=cmap(i/classifier_num), label='SVM ROC curve (area = %0.2f)' % roc_auc[i])
    i += 1

    fpr[i], tpr[i], _ = roc_curve(prediction['MLP_predict'], dataset['y_test'])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, c=cmap(i/classifier_num), label='MLP ROC curve (area = %0.2f)' % roc_auc[i])
    i += 1

    
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic example\n[' + prediction['name'] + ' Dataset]')
plt.legend(loc="lower right")
plt.show()
