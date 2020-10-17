################################################################################
### Q2-2 Multi-Layer Perceptron on breat cancer dataset ########################
###      Show how the weights from each feature to each neuron are organized ###
################################################################################

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

### 데이터셋 준비 ###
cancer=load_breast_cancer()	# 유방암 데이터셋 로딩

print("max:\n{}".format(cancer.data.max(axis=0)))	# 데이터셋의 Feature의 최대값 출력

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

### Multi-Layer Perceptron 준비 및 학습 ###

mlp = MLPClassifier(random_state=42)	# Neuron 100개인 Hidden Layer 1개로 구성된 MLP 준비
					# Random state 42로 설정

mlp.fit(X_train, y_train)	# Train 데이터셋에 대해 학습 수행

print("train accuracy: {:.2f}".format(mlp.score(X_train, y_train)))	# Train 데이터에 대한 Prediction 정확도 출력
print("test accuracy: {:.2f}".format(mlp.score(X_test, y_test)))	# Test 데이터에 대한 Prediction 정확도 출력

### 데이터셋 표준화 ###
# Train 데이터셋의 각 Feature에 대한 평균 산출
mean_on_train = X_train.mean(axis=0)

# Train 데이터셋의 각 Feature에 대한 표준편차 산출
std_on_train = X_train.std(axis=0)

# Train 데이터셋에 대해 각 Feature를 평균으로 빼고, 표준편차로 나눠서
# 평균 0, 표준편차 1인 z-score 형태로 데이터셋을 표준화함
X_train_scaled = (X_train - mean_on_train) / std_on_train

# Test 데이터셋에 대해서 z-score 표준화 수행
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0)	# Neuron 100개인 Hidden Layer 1개로 구성된 MLP 준비
					# Random state 0으로 설정

mlp.fit(X_train_scaled, y_train)	# 표준화된 Train 데이터셋에 대해 학습 수행

print("train accuracy: {:.3f}".format(mlp.score(X_train_scaled, y_train)))	# 표준화된 Train 데이터에 대한 Prediction 정확도 출력
print("test accuracy: {:.3f}".format(mlp.score(X_test_scaled, y_test)))		# 표준화된 Test 데이터에 대한 Prediction 정확도 출력

mlp = MLPClassifier(max_iter=1000, random_state=0)	# Neuron 100개인 Hidden Layer 1개로 구성된 MLP 준비
							# Regression 최대 횟수 1000으로 설정
mlp.fit(X_train_scaled, y_train)			# 표준화된 Train 데이터셋에 대해 학습 수행

print("train accuracy: {:.3f}".format(mlp.score(X_train_scaled, y_train)))	# 표준화된 Train 데이터에 대한 Prediction 정확도 출력
print("test accuracy: {:.3f}".format(mlp.score(X_test_scaled, y_test)))		# 표준화된 Test 데이터에 대한 Prediction 정확도 출력

mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)	# Neuron 100개인 Hidden Layer 1개로 구성된 MLP 준비
								# Regression 최대 횟수 1000으로 설정 / L2 Regularization Penalty 1로 설정
mlp.fit(X_train_scaled, y_train)				# 표준화된 Train 데이터셋에 대해 학습 수행

print("train accuracy: {:.3f}".format(mlp.score(X_train_scaled, y_train)))	# 표준화된 Train 데이터에 대한 Prediction 정확도 출력
print("test accuracy: {:.3f}".format(mlp.score(X_test_scaled, y_test)))		# 표준화된 Test 데이터에 대한 Prediction 정확도 출력

'''
*** 표준화된 데이터셋을 사용하여 Scaling 문제를 해결함으로서 정확도를 높일 수 있음 ***
'''

# 표준화된 데이터셋으로 학습된 MLP의 Feature-Neuron 사이의 개별 Weight값을 Hidden Layer의 Neuron별로 2D 그래프로 표현함 
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("hidden unit")
plt.ylabel("input")
plt.colorbar()
plt.title('Weight values from each feature to each neuron')
plt.show()
