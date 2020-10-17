#########################################################
### Q1-1 : Iris Classification using Perceptron Class ###
#########################################################
import numpy as np

### Perceptron Class 생성 ###
class Perceptron(object):

	# Perceptron 생성/초기화 함수
	def __init__(self, eta=0.01, n_iter=50, random_state=1):
		'''
		eta : Learning rate
		n_iter : Number of Iteration for classification in single NN layer
		random_state : Number of perceptrons/neurons used in single NN layer
		'''
		self.eta = eta                      # Learning Rate 선언
		self.n_iter = n_iter                # Regression 반복 횟수 선언
		self.random_state = random_state    # 단일 Perceptron/Neuron에 입력될 최초 Weight (Seed)의 초기화 개수 선언

	# Perceptron의 Regression 함수
	def fit(self, X, y):
		'''
		X : Perceptron 입력 데이터 ('데이터셋의 Feature 데이터' 또는 '이전 Layer의 Perceptron에서 전달된 Output')
		y : 데이터셋의 Label/Target 데이터
		'''
		rgen = np.random.RandomState(self.random_state)
		# Perceptron/Neuron에 입력되는 Weight 개수만큼 Weight 초기값(Seed)을 생성하기 위한 난수 발생기
		# Reproducibility를 보장하기 위해 random_state를 설정하여 매번 동일한 형태의 난수를 발생하여 최초 Weight (Seed)를 설정하게함

		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
		# 입력 weight의 초기값을 Normal Distribution으로 구성된 랜덤한 값으로 설정함

		self.errors_ = []   # 매 Regression마다 발생하는 Error값 저장

		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):	# 전체 데이터셋에 대해서 Regression 수행
				
				# weight 대비 Output Error/Loss의 변화량/미분값(dL/dw)에 비례하여 각 Perceptron의 Weight를 Update함
				# 이를 위해 weight 대비 Output Error/Loss의 미분값(dL/dw)을 Learning Rate만큼 곱해서 Weight를 업데이트하여 Gradient Descent를 구현함
				
				# weight 대비 Output Error/Loss (dL/dw)는 Chain Rule에 의해 
				# Loss 함수의 미분값 * Activation 함수의 미분값 * 이전 Layer의 Output (Activation을 거친 Weighted Sum)으로 구할 수 있음

				update = self.eta * (target - self.predict(xi))		
				# 현재 Perceptron에서 사용하는 Activation Function이 Step Function이기 때문에 (predict 함수 참고) 
				# 미분한 값이 x=0에서 1인 Impulse Function임. 그러므로 Activation Function의 미분값으로 1이 사용되어 곱해짐.
				
				self.w_[1:] += update * xi	
				# Perceptron의 입력 Weight (이전 Neuron의 Output에 곱해져서 전달되는 비율)를 입력 데이터, Weight에 따른 Error/Loss 변화량, 
				# Learning Rate를 반영하여 더하여 누적 업데이트함 (Gradient Descent에 의한 Weight 업데이트)

				self.w_[0] += update
				# Perceptron의 입력 Bias (이전 Neuron의 Output에 더해져서 전달되는 양)는 미분하면 0이기에 Weight에 따른 Error/Loss 변화량과 
				# Learning Rate만 반영하여 더하여 누적 업데이트함 (Gradient Descent에 의한 Bias 업데이트)

				errors += int(update != 0.0)	# Error값이 0이 아닌 경우 (Output과 Prediction이 일치하지 않는 경우)에만 Error값을 누적함


			self.errors_.append(errors)

		return self

	# Perceptron의 Weighted Sum 연산 함수
	def net_input(self, X):

		return np.dot(X, self.w_[1:]) + self.w_[0]	# Matrix간 Dot Product를 통한 Weighted Sum과 Bias (0번 Weight)를 더함

	# Prediction 함수
	def predict(self, X):
			
		# np.where를 사용하여 Step Function의 Activation Function을 구현함
		# Weighted Sum이 0보다 큰 경우 1, 작은 경우 -1로 나눌 수 있는 Step Function을 통해 Perceptron의 최종 Output의 출력함
		return np.where(self.net_input(X) >= 0.0, 1, -1)


### Iris 데이터셋을 이용한 Perceptron 훈련 ###
import pandas as pd

# 데이터셋 로딩
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
		 'machine-learning-databases/iris/iris.data', header=None)

# 데이터셋 끝단 5개 출력
print(df.tail())

import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 0 ~ 3번 Column : Feature 데이터
# 데이터셋 4번 Column : Label/Target 데이터
y = df.iloc[0:100, 4].values	# Dataset에서 0 ~ 99번 데이터의 Label/Target (4번 Column)만 추출
				# Dataset에서 0 ~ 99번 데이터의 Label/Target은 setosa와 versicolor로만 구성됨
# setosa와 veriscolor에 대해서만 Classification 수행을 위해 Target에 대해 숫자 Labeling 수행
y = np.where(y == 'Iris-setosa', -1, 1)	# 데이터셋에서 Label이 Iris-setosa인 경우 -1로 지정
					# 데이터셋에서 Label이 Iris-versicolor인 경우 1로 지정

X = df.iloc[0:100, [0, 2]].values	# Dataset에서 0 ~ 99번 데이터의 0번 Column(꽃받침 길이)과 2번 Column(꽃잎 길이) 추출

# Feature 데이터에 대해 Label의 분포를 확인하기 위해 Scatter 그래프 작성
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')		# 0 ~ 49번 setosa Label의 꽃받침 길이와 꽃잎 길이 대비 분포 작성
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')	# 50 ~ 99번 setosa Label의 꽃받침 길이와 꽃잎 길이 대비 분포 작성

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()

# Perceptron 준비 및 학습

ppn = Perceptron(eta=0.1, n_iter=10)
# Regression Learning Rate = 0.1, 반복 횟수 = 10회로 설정된 Perceptron 선언

ppn.fit(X, y)
# Iris 데이터셋의 0 ~ 99번 데이터의 0번 Column(꽃받침 길이)과 2번 Column(꽃잎 길이)를 
# Feature로 사용하는 상황에서 Label/Target (4번 Column)을 Classification 하도록 Regression 학습 수행

print(ppn.errors_)	# Perceptron 학습 결과의 Iteration마다 발생한 Error값 출력

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')	# 매 학습 Iteration (Epoch)마다 발생한 Error값의 그래프 작성
plt.xlabel('Epochs')
plt.ylabel('Number of errors')

plt.show()

### Perceptron 훈련 결과 Decision Boundary로 표현 ###
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, classifier, resolution=0.02):
	# 그래프 상 각 데이터의 Label과 영역을 표시하기 위한 Marker와 Colormap 선언
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])	# 각 Label별로 리스트 상 순서에 따라 고유의 색이 사용되게함

	# Decision Boundary 생성
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1	# Dataset의 0번 Column 데이터 범위
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 0].max() + 1	# Dataset의 1번 Column 데이터 범위

	# np.meshgrid를 사용하여 Dataset의 0번, 1번 데이터 범위로 구성된 2D Matrix 생성
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	# Dataset의 0번, 1번 데이터를 각각 Column-wise로 합쳐서 Classifier에게 전달하여 Prediction 수행함
	# Dataset 전 범위에 걸쳐서 Prediction을 수행함
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)	# Prediction 결과를 입력 데이터 형태와 동일하게 재구성함

	plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)	
	# 데이터 전 범위에 대한 Prediction 결과에서 1인 경우 파란색, -1인 경우 빨간색으로 칠해서 Decision Boundary에 의한 영역분할을 나타냄
	plt.xlim(xx1.min(), xx1.max())	# 그래프 X 범위 설정
	plt.ylim(xx2.min(), xx2.max())	# 그래프 y 범위 설정

	# 주어진 Iris 데이터셋에 대한 Prediction 결과를 Scatter로 추가로 그려넣음
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], 
			    y=X[y == cl, 1],
			    alpha=0.8, 
			    c=colors[idx],
			    marker=markers[idx], 
			    label=cl, 
			    edgecolor='black')

plot_decision_boundary(X, y, classifier=ppn)
# Perceptron을 이용한 Prediction 결과에 대한 Decision Boundary를 그래프로 그림

plt.xlabel('sepal length [cm]')		# 그래프 X축 설명
plt.ylabel('petal length [cm]')		# 그래프 y축 설명
plt.legend(loc='upper left')

plt.show()

