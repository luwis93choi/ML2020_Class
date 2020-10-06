###########################################
### Q1. Basic KNN Algorithm Development ###
###########################################

### 연습용 데이터셋 준비 / 데이터셋 데이터 형 : [x, y, Label] ###
dataset = [[2.7810836,   2.550537003, 0],
	   [1.465489372, 2.362125076, 0],
	   [3.396561688, 4.400293529, 0],
	   [1.38807019,  1.850220317, 0],
	   [3.06407232,  3.005305973, 0],
	   [7.627531214, 2.759262235, 1],
	   [5.332441248, 2.088626775, 1],
	   [6.922596716, 1.77106367,  1],
	   [8.675418651,-0.242068655, 1],
	   [7.673756466, 3.508563011, 1]]

from math import sqrt

# 데이터셋의 데이터 (1x3 벡터) 간 Euclidean Distance를 계산하는 함수
# 데이터셋 데이터 형식: [x, y, Label]
def euclidean_distance(row1, row2):
    
    distance = 0.0
    
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i])**2
    
    return sqrt(distance)
    
row0 = [3, 3]   # 테스트용 데이터

for row in dataset:
    # row0 변수와 연습용 데이터셋 내 모든 데이터간 Euclidean Distance를 연산함
    distance = euclidean_distance(row0, row)
    print(distance)

# 데이터셋 내에서 입력 데이터와 Euclidean Distance가 가까운 K개의 데이터(Neighbor)를 찾는 함수
def get_neighbors(train, test_row, num_neighbors):
	
	distances = list()
	
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))

	# 데이터셋 내에서 입력 데이터와의 Euclidean Distance 연산 결과값을 오름차순으로 정리해서 Distance가 작은 값이 앞에 배치되게함
	distances.sort(key=lambda tup:tup[1])

	neighbors = list()

	# 입력 데이터와 Euclidean Distance가 작은 순서대로 데이터셋 데이터 K개를 선정함
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])

	return neighbors

neighbors = get_neighbors(dataset, row0, 3)	# row0 테스트 데이터와 Euclidean Distance가 가까운 데이터 3개를 선정함 (KNN neighbor=3 수행)

for neighbor in neighbors:
	print(neighbor)		# row0에 대해 Euclidean Distance가 가까운 Neighbor 데이터 3개를 출력함

### K-Nearest Neighbor Classification 함수 ###
def predict_classification(train, test_row, num_neighbors):

	# KNN 1단계 : 입력 데이터에 대해 Distance가 가까운 데이터 K개를 데이터셋에서 선정함
	neighbors = get_neighbors(train, test_row, num_neighbors)
	for neighbor in neighbors:
		print(neighbor)

	# KNN 2단계 : 입력 데이터 기준으로 선정된 Neighbor의 Label을 확인해서 과반수인 Label의 입력 데이터에 대한 결과를 Prediction으로 정의함 (Majority Voting)
	output_values = [row[-1] for row in neighbors]		# Neighbor에 대한 Label을 모두 수집함
								# 음수 Index값 : 리스트의 Index를 오른쪽 방향에서 선택함

	prediction = max(set(output_values), key=output_values.count)	# Neighbor에 대한 Majority Voting을 수행함
									# Neighbor의 Label 중에서 과반수로 선정된 것을 Prediction으로 정의함

	return prediction

row0 = [3, 3, 0]	# 테스트 데이터

prediction = predict_classification(dataset, row0, 3)	# 테스트 데이터에 대해 Neighbor 3개를 선정하는 KNN Classification 수행함

print('Expected %d, Got %d.' % (row0[-1], prediction))	# 테스트 데이터의 Label와 KNN Classification의 Prediction 결과를 비교함

##########################################################
### Q1. Breast Cancer Classification Example using KNN ###
##########################################################

### 데이터셋 준비 및 확인 ###
from sklearn.datasets import load_breast_cancer		# 유방암 데이터셋 라이브러리 준비

breast_cancer_data = load_breast_cancer()		# 유방암 데이터셋 준비

print(breast_cancer_data.keys())			# 유방암 데이터셋의 Attribute (구성요소) 출력

print(breast_cancer_data.target_names)			# 유방암 데이터셋의 Target/Label에 대한 내용 출력 (유방암 악성 여부)

import pandas as pd
df_data = pd.DataFrame(breast_cancer_data.data)			# 유방암 데이터셋의 Feature 데이터를 Pandas DataFrame 형태로 전환
df_labels = pd.DataFrame(breast_cancer_data.target)		# 유방암 데이터셋의 Target/Label을 Pandas DataFrame 형태로 전환

print(df_data.head())		# 유방암 데이터셋의 Feature 데이터 최초 5개를 출력함
				# 유방암 데이터셋의 30개의 Feature로 구성되어있다는 것을 알 수 있음

print(df_labels.head())		# 유방암 데이터셋의 Target/Label 데이터 최초 5개를 출력함

print(df_data.describe())	# 유방암 데이터셋의 Feature 데이터에 대한 통계학적 정보를 출력함

### 데이터셋 정규화 ###
# Min-Max 정규화 함수
def min_max_normalize(lst):
	normalized = []
	for value in lst:
		normalized_num = (value - min(lst)) / (max(lst) - min(lst))	# 리스트에 있는 값을 Min-Max 정규화를 수행하여 모든 값을 0 ~ 1 사이의 값으로 정리함
		normalized.append(normalized_num)
	return normalized

# 각 Column Feature 데이터에 대해 Min-Max 정규화를 수행함
# 각 Feature마다 다른 단위 체계를 사용하기 때문에 Scaling에 의해 데이터의 분포가 단위별로 다르게 나타나는 현상이 발생함
# Scaling에 의한 부정적인 효과를 최소화 시키기 위해 정규화를 통해 모든 데이터를 0 ~ 1 사이 범위로 재정의하여 단위 체계를 비율 체계로 통일화시킴
for x in range(len(df_data.columns)):

	# 각 Feature Column의 데이터는 정규화되어 다시 데이터셋에 저장됨
	df_data[x] = min_max_normalize(df_data[x])

print(df_data.describe())	# 정규화된 유방암 데이터셋의 Feature 데이터에 대한 통계학적 정보를 출력함

print(df_data)			# 정규화된 유방암 데이터셋의 Feature 데이터를 출력함

### 데이터셋 Training/Validation 분리 ###
from sklearn.model_selection import train_test_split

# 데이터셋 데이터를 Training 80% : Validation 20%로 분리함
training_data, validation_data, training_labels, validation_labels = train_test_split(df_data, df_labels, test_size=0.2, random_state=100)

print(len(training_data))		# Training 데이터 개수 출력
print(len(validation_data))		# Validation 데이터 개수 출력
print(len(training_labels))		# Training Label 데이터 개수 출력
print(len(validation_labels))		# Validation Label 데이터 개수 출력

### KNN Classification을 통한 Prediction ###
# 유방암 데이터셋에 대해 KNN Classifcation 적용
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)	# Neighbor 3개를 선정하는 KNN Classifier 생성

classifier.fit(training_data, training_labels)	
# *** KNN Classifier가 Learning이나 Training이 필요하지 않는 Lazy Classifier임에도 Fit을 하는 이유 = '탐색 효율 증가' *** 
#     : NxM 벡터로 구성된 데이터셋 내에서 전체 탐색이 아닌 효과적인 방식으로 K개의 Neighbor를 찾기 위해서는
#       데이터셋의 데이터를 탐색에 용의한 자료구조로 재정리할 필요가 있음
#       Fit을 통해서 현재 데이터셋에 대해 어떤 자료구조를 사용해서 정리할 것인지 결정하고
#       결정된 자료구조로 데이터를 배치해서 K개의 Neighbor를 보다 효율적으로 찾아냄
#       scikit-learn의 KNeighborsClassifier는 Ball Tree, KD Tree, Brute Force 등 여러 자료구조를 사용하여 Neighbor 탐색 수행
#       (출처 : https://stats.stackexchange.com/questions/349842/why-do-we-need-to-fit-a-k-nearest-neighbors-classifier)

print(classifier.score(validation_data, validation_labels))	# KNN Classifier에 대한 성능을 검증하기 위해 Test Data에 대한 Label 추정 정확도 확인

# K의 값을 바꿔가면서 KNN의 정확도/성능을 확인함
import matplotlib.pyplot as plt
k_list = range(1, 101)
accuracies = []

# K의 값을 0에서 100까지 바꿔가면서 KNN Classification 수행함
for k in k_list:
	# 0에서 100 사이의 K값을 바꿔가면서 유방암 데이터셋에 대한 Fitting / Classification 수행
	classifier = KNeighborsClassifier(n_neighbors=k)	
	classifier.fit(training_data, training_labels)
	
	# 0에서 100 사이의 K값을 바꿔가면서 KNN의 Classification 정확도를 저장함
	accuracies.append(classifier.score(validation_data, validation_labels))

# K값이 변하면서 Classification 정확도가 어떻게 변하는지 확인하여 최대 정확도를 내는 K값을 선정할 수 있음
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
