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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

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
		         {'name' : 'Standardized', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None},
		         {'name' : 'PCA-Original', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None},
		         {'name' : 'PCA-Standardized', 'X_train' : None, 'y_train' : None, 'X_test' : None, 'y_test' : None}]

# 비표준화 데이터셋, 표준화 데이터셋을 각각 Training 70%, Test 30%로 분리함
# 비표준화 데이터셋 Training/Validation 70%, Test 30% 분리
(dataset_group[0]['X_train'], dataset_group[0]['X_test'], 
 dataset_group[0]['y_train'], dataset_group[0]['y_test']) = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.3, random_state=23)

# 표준화 데이터셋 Training/Validation 70%, Test 30% 분리
(dataset_group[1]['X_train'], dataset_group[1]['X_test'],
 dataset_group[1]['y_train'], dataset_group[1]['y_test']) = train_test_split(breast_cancer_data_std, breast_cancer_label, test_size=0.3, random_state=23)

# 각 데이터셋 종류와 Classifier 종류에 따른 정확도를 저장하기 위한 자료 구조 생성
accuracy = [{'name' : 'Original', 'Kmeans_acc' : None, 'GMM_acc' : None, 'Agglo_acc' : None, 'DBSCAN_acc' : None},
	        {'name' : 'Standardized', 'Kmeans_acc' : None, 'GMM_acc' : None, 'Agglo_acc' : None, 'DBSCAN_acc' : None},
	        {'name' : 'PCA-Original', 'Kmeans_acc' : None, 'GMM_acc' : None, 'Agglo_acc' : None, 'DBSCAN_acc' : None},
	        {'name' : 'PCA-Standardized', 'Kmeans_acc' : None, 'GMM' : None, 'Agglo_acc' : None, 'DBSCAN_acc' : None}]

predictions = [{'name' : 'Original', 'Kmeans_predict' : None, 'GMM_predict' : None, 'Agglo_predict' : None, 'DBSCAN_predict' : None},
	           {'name' : 'Standardized', 'Kmeans_predict' : None, 'GMM_predict' : None, 'Agglo_predict' : None, 'DBSCAN_predict' : None},
	           {'name' : 'PCA-Original', 'Kmeans_predict' : None, 'GMM_predict' : None, 'Agglo_predict' : None, 'DBSCAN_predict' : None},
	           {'name' : 'PCA-Standardized', 'Kmeans_predict' : None, 'GMM_predict' : None, 'Agglo_predict' : None, 'DBSCAN_predict' : None}]

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
pca_original = PCA(n_components=2, random_state=42).fit(dataset_group[0]['X_train'])
pca_std = PCA(n_components=5, random_state=42).fit(dataset_group[1]['X_train'])

dataset_group[2]['X_train'] = pca_original.transform(dataset_group[0]['X_train'])
dataset_group[2]['y_train'] = dataset_group[0]['y_train']

dataset_group[2]['X_test'] = pca_original.transform(dataset_group[0]['X_test'])
dataset_group[2]['y_test'] = dataset_group[0]['y_test']

dataset_group[3]['X_train'] = pca_std.transform(dataset_group[1]['X_train'])
dataset_group[3]['y_train'] = dataset_group[1]['y_train']

dataset_group[3]['X_test'] = pca_std.transform(dataset_group[1]['X_test'])
dataset_group[3]['y_test'] = dataset_group[1]['y_test']

PCA_breast_cancer_data = pca_original.transform(breast_cancer_data)
PCA_breast_cancer_data_std = pca_original.transform(breast_cancer_data_std)

################################################################################################################
### 04 Clustering-based Classification #########################################################################
################################################################################################################

classifier_num = 0
for dataset in dataset_group:

    print('{}'.format(dataset['name']))

    ### K-Means / Evaluation with Test Set ###
    kmeans = KMeans(n_clusters=len(np.unique(breast_cancer_label)), random_state=42).fit(dataset['X_train'])

    kmeans_predictions = kmeans.predict(dataset['X_test'])
    kmeans_predictions_invert = 1-kmeans_predictions

    acc_original = metrics.accuracy_score(dataset['y_test'], kmeans_predictions)
    acc_inverted = metrics.accuracy_score(dataset['y_test'], kmeans_predictions_invert)

    kmeans_prediction = None
    if acc_original >= acc_inverted:

        kmeans_acc = acc_original
        kmeans_prediction = kmeans_predictions

    else:

        kmeans_acc = acc_inverted
        kmeans_prediction = kmeans_predictions_invert

    print('Kmeans Acc : {}'.format(kmeans_acc))

    accuracy['name'==dataset['name']]['Kmeans_acc'] = kmeans_acc
    predictions['name'==dataset['name']]['Kmeans_predict'] = kmeans_prediction

    confusion_Mat = confusion_matrix(dataset['y_test'], kmeans_prediction)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ["{0:.2%}".format(value) for value in confusion_Mat.flatten()/np.sum(confusion_Mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_Mat/np.sum(confusion_Mat), annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix of Best K-Means Model\n[' + dataset['name'] + ' Dataset]')
    plt.show()

    classifier_num += 1
    
    ### GMM / Evaluation with Test Set ###
    GMM = GaussianMixture(n_components=len(np.unique(breast_cancer_label)), n_init=100, random_state=42).fit(dataset['X_train'])

    GMM_predictions = GMM.predict(dataset['X_test'])
    GMM_predictions_invert = 1-GMM_predictions

    acc_original = metrics.accuracy_score(dataset['y_test'], GMM_predictions)
    acc_inverted = metrics.accuracy_score(dataset['y_test'], GMM_predictions_invert)

    GMM_prediction = None
    if acc_original >= acc_inverted:

        GMM_acc = acc_original
        GMM_prediction = GMM_predictions

    else:

        GMM_acc = acc_inverted
        GMM_prediction = GMM_predictions_invert

    print('GMM Acc : {}'.format(GMM_acc))

    accuracy['name'==dataset['name']]['GMM_acc'] = GMM_acc
    predictions['name'==dataset['name']]['GMM_predict'] = GMM_prediction

    confusion_Mat = confusion_matrix(dataset['y_test'], GMM_prediction)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ["{0:.2%}".format(value) for value in confusion_Mat.flatten()/np.sum(confusion_Mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_Mat/np.sum(confusion_Mat), annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix of Best GMM Model\n[' + dataset['name'] + ' Dataset]')
    plt.show()

    classifier_num += 1
    
    ### Agglomerative Clustering ###
    AGG = AgglomerativeClustering(n_clusters=len(np.unique(breast_cancer_label)))

    if dataset['name'] == 'Original':
        AGG_predictions = AGG.fit_predict(breast_cancer_data)
    
    elif dataset['name'] == 'PCA-Original':
        AGG_predictions = AGG.fit_predict(PCA_breast_cancer_data)
    
    elif dataset['name'] == 'Standardized':
        AGG_predictions = AGG.fit_predict(breast_cancer_data_std)
    
    elif dataset['name'] == 'PCA-Standardized':
        AGG_predictions = AGG.fit_predict(PCA_breast_cancer_data_std)
    
    AGG_predictions_invert = 1-AGG_predictions

    acc_original = metrics.accuracy_score(breast_cancer_label, AGG_predictions)
    acc_inverted = metrics.accuracy_score(breast_cancer_label, AGG_predictions_invert)

    AGG_prediction = None
    if acc_original >= acc_inverted:

        AGG_acc = acc_original
        AGG_prediction = AGG_predictions

    else:

        AGG_acc = acc_inverted
        AGG_prediction = AGG_predictions_invert

    print('Agglomerative Clustering Acc : {}'.format(AGG_acc))

    accuracy['name'==dataset['name']]['Agglo_acc'] = AGG_acc
    predictions['name'==dataset['name']]['Agglo_predict'] = AGG_prediction

    confusion_Mat = confusion_matrix(breast_cancer_label, AGG_prediction)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ["{0:.2%}".format(value) for value in confusion_Mat.flatten()/np.sum(confusion_Mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_Mat/np.sum(confusion_Mat), annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix of Best Agglomerative Clustering Model\n[' + dataset['name'] + ' Dataset]')
    plt.show()

    classifier_num += 1
    
    ### DBSCAN ###
    best_dbscan_acc = 0.0
    best_EPS = None
    best_minPts = None
    dbscan_prediction = None
    for EPS in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        
        for MINPts in range(1, 51):

            dbscan = DBSCAN(eps=EPS, min_samples=MINPts, metric='euclidean')

            if dataset['name'] == 'Original':
                dbscan_predictions = dbscan.fit_predict(breast_cancer_data)
            
            elif dataset['name'] == 'PCA-Original':
                dbscan_predictions = dbscan.fit_predict(PCA_breast_cancer_data)
            
            elif dataset['name'] == 'Standardized':
                dbscan_predictions = dbscan.fit_predict(breast_cancer_data_std)
            
            elif dataset['name'] == 'PCA-Standardized':
                dbscan_predictions = dbscan.fit_predict(PCA_breast_cancer_data_std)

            dbscan_predictions_invert = 1 + dbscan_predictions

            acc_original = metrics.accuracy_score(breast_cancer_label, dbscan_predictions)
            acc_inverted = metrics.accuracy_score(breast_cancer_label, dbscan_predictions_invert)

            if acc_original >= acc_inverted:

                dbscan_acc = acc_original

                if dbscan_acc >= best_dbscan_acc:

                    best_dbscan_acc = dbscan_acc
                    best_EPS = EPS
                    best_minPts = MINPts
                    dbscan_prediction = dbscan_predictions

            else:

                dbscan_acc = acc_inverted

                if dbscan_acc >= best_dbscan_acc:

                    best_dbscan_acc = dbscan_acc
                    best_EPS = EPS
                    best_minPts = MINPts
                    dbscan_prediction = dbscan_predictions_invert

    print('DBSCAN Clustering Acc : {}'.format(best_dbscan_acc))
    print('Best DBSCAN Params : eps = {} | minPts = {}'.format(best_EPS, best_minPts))

    accuracy['name'==dataset['name']]['DBSCAN_acc'] = best_dbscan_acc
    predictions['name'==dataset['name']]['DBSCAN_predict'] = dbscan_prediction

    confusion_Mat = confusion_matrix(breast_cancer_label, dbscan_prediction)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ["{0:.2%}".format(value) for value in confusion_Mat.flatten()/np.sum(confusion_Mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_Mat/np.sum(confusion_Mat), annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix of Best DBSCAN Model\n[' + dataset['name'] + ' Dataset]')
    plt.show()

    classifier_num += 1
    
    print('\n')

    print(predictions['name'==dataset['name']])

print(predictions)

cmap = plt.cm.rainbow
i = 0
for prediction, dataset in zip(predictions, dataset_group):

    fpr = [0 for i in range(int(classifier_num))]
    tpr = [0 for i in range(int(classifier_num))]
    roc_auc = [0 for i in range(int(classifier_num))]
    
    fpr[i], tpr[i], _ = roc_curve(prediction['Kmeans_predict'], dataset['y_test'])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, c=cmap(i/classifier_num), label='K-Means ROC curve (area = %0.2f)' % roc_auc[i])
    i += 1

    fpr[i], tpr[i], _ = roc_curve(prediction['GMM_predict'], dataset['y_test'])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, c=cmap(i/classifier_num), label='GMM ROC curve (area = %0.2f)' % roc_auc[i])
    i += 1

    fpr[i], tpr[i], _ = roc_curve(prediction['Agglo_predict'], breast_cancer_label)
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, c=cmap(i/classifier_num), label='Agglomerative Clustering ROC curve (area = %0.2f)' % roc_auc[i])
    i += 1

    fpr[i], tpr[i], _ = roc_curve(prediction['DBSCAN_predict'], breast_cancer_label)
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, c=cmap(i/classifier_num), label='DBSCAN ROC curve (area = %0.2f)' % roc_auc[i])
    i += 1

    
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic example\n[' + prediction['name'] + ' Dataset]')
plt.legend(loc="lower right")
plt.show()