#########################################################
### Q2-1 Multi-Layer Perceptron on clustering dataset ###
#########################################################

from sklearn.neural_network import MLPClassifier	# Multi-Layer Perceptron 기반 Classifier
from sklearn.datasets import make_moons			# Clustering 데이터셋
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt

### 데이터셋 준비 ###
# Noise가 첨가된 100개의 Clustering 데이터를 데이터셋으로 준비 
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

### Hidden Layer의 Neuron 구성에 따른 MLP Classification 결과 비교 ###
# Neuron 100개인 Hidden Layer 1개로 구성된 Mutli-Layer Perceptron 생성
# Weight Optimization을 위해 Quasi-Newton Optimizer 사용
mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)	# Train 데이터에 대해 학습 수행

mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)	# Train 데이터의 영역별로 나눠서 색칠함
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)		# Train 데이터를 Scatter Plot으로 표현함

# MLP를 이용한 Classification 결과를 그래프로 나타냄
plt.title("n_hidden=[{}]\nrandom_state={}\nL2 penalty(alpha)={}\nWeight Optimization Solver={}".format(100, 0, 0.0001, 'Quasi-Newton Optimizer'))
plt.xlabel('zero')
plt.ylabel('one')
plt.show()

# Neuron 10개인 Hidden Layer 1개로 구성된 Mutli-Layer Perceptron 생성
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])

# Train 데이터에 대해 학습 수행
mlp.fit(X_train, y_train)

mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)	# Train 데이터의 영역별로 나눠서 색칠함
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)		# Train 데이터를 Scatter Plot으로 표현함

# MLP를 이용한 Classification 결과를 그래프로 나타냄
plt.title("n_hidden=[{}]\nrandom_state={}\nL2 penalty(alpha)={}\nWeight Optimization Solver={}".format(10, 0, 0.0001, 'Quasi-Newton Optimizer'))
plt.xlabel('zero')
plt.ylabel('one')
plt.show()

'''
*** Hidden Layer에 더 많은 Neuron을 사용할수록 Decision Boundary를 좀 더 Non-Linear하게 표현할 수 있음 ***
*** Non-lienar한 분포에 대해 Neuron이 많이 사용할수록 좀 더 부드러운 Non-linear Decision Boundary 생성 가능 ***
'''

### Hidden Layer의 Neuron 구성, Hidden Layer 사용 개수, Regularization L2 Penalty, Random State에 따른 MLP Classification 결과 비교
fig, axes = plt.subplots(2, 4, figsize=(20, 8))		# 2x4로 구성된 Subplot 구조 생성

								# Hidden Layer 2개를 사용하는 상황에서
for axx, n_hidden_nodes in zip(axes, [10, 100]):		# Hidden Layer에 사용할 Neuron의 개수를 변경하면서
	for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):	# Regularization L2 Penalty 비율을 변경하면서 
								# MLP Classification 성능을 확인함
		
		# 변경된 Neuron 개수와 L2 Penalty를 반영하여 MLP 생성
		mlp = MLPClassifier(solver='lbfgs', random_state=0,
				    hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], 
				    alpha=alpha)

		mlp.fit(X_train, y_train)	# Train 데이터에 대해 학습 수행

		mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)	# Train 데이터의 영역별로 나눠서 색칠함
		mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)		# Train 데이터를 Scatter Plot으로 표현함
	
		# MLP를 이용한 Classification 결과를 그래프로 나타냄
		ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))

fig, axes = plt.subplots(2, 4, figsize=(20, 8))		# 2x4로 구성된 Subplot 구조 생성


					# Neuron 100개로 각각 구성된 Hidden Layer 2개를 사용하는 상황에서
for i, ax in enumerate(axes.ravel()):	# Random State를 변경하면서
					# MLP Classification 성능을 확인함

	# 변경된 Random State를 반영하여 MLP 생성
	mlp = MLPClassifier(solver='lbfgs', random_state=i, hidden_layer_sizes=[100, 100])

	mlp.fit(X_train, y_train)	# Train 데이터에 대해 학습 수행

	mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)	# Train 데이터의 영역별로 나눠서 색칠함
	mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)		# Train 데이터를 Scatter Plot으로 표현함

	# MLP를 이용한 Classification 결과를 그래프로 나타냄
	ax.set_title("n_hidden=[{}, {}]\nrandom_state={}".format(n_hidden_nodes, n_hidden_nodes, i))

plt.show()


'''
*** Hidden Layer에 사용되는 Neuron이 증가할수록 Non-Linear한 Decision Boundary를 좀 더 부드럽게 그릴 수 있음 ***
*** alpha가 커질수록 Regularization Penalty가 강하게 나타나서 Decision Boundary가 데이터에 Overfitting되는 것을 약화시킴 ***
*** Random state가 달라질 수록 Classification 결과가 달라지기에 Random state를 고정하여 Reproducibility를 보장해야함 ***
'''
