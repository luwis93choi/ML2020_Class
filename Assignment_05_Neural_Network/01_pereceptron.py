############################################################
### Q1 : Iris Classification using 1 Layer of Perceptron ###
############################################################
import numpy as np

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
		self.random_state = random_state    # 단일 Neural Network Layer에 사용될 Perceptron/Neuron 개수 선언

	# Perceptron의 Regression 함수
	def fit(self, X, y):
		'''
		X : Perceptron 입력 데이터 ('데이터셋의 Feature 데이터' 또는 '이전 Layer의 Perceptron에서 전달된 Output')
		y : 데이터셋의 Label/Target 데이터
		'''
		rgen = np.random.RandomState(self.random_state)
		# Layer에 입력되는 Perceptron/Neuron의 개수만큼 weight 초기값을 생성하기 위한 난수 발생기

		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
		# Normal Distribution으로 입력 weight의 초기값을 설정함

		self.errors_ = []   # 매 Regression마다 발생하는 Error값 저장

		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):	# 전체 데이터셋에 대해서 Regression 수행
				
				# weight 대비 Output Error/Loss의 변화량/미분값(dL/dw)에 비례하여 각 Perceptron의 Weight를 Update함
				# 이를 위해 weight 대비 Output Error/Loss의 미분값(dL/dw)을 Learning Rate만큼 곱해서 Weight를 업데이트하여 Gradient Descent를 구현함
				# weight 대비 Output Error/Loss (dL/dw)는 Chain Rule에 의해 Loss 함수의 미분값 * Activation Function의 미분값 * 이전 Layer의 Output (Activation을 거친 Weighted Sum)으로 구할 수 있음

				update = self.eta * (target - self.predict(xi))		
				# 현재 Perceptron에서 사용하는 Activation Function이 Step Function이기 때문에 (predict 함수 참고) 미분한 값이 x=0에서 1인 Impulse Function임. 그러므로 Activation Function의 미분값으로 1이 사용되어 곱해짐.
				self.w_[1:] += update * xi	
				# Perceptron의 입력 Weight (이전 Neuron의 Output에 곱해져서 전달되는 비율)를 입력 데이터, Weight에 따른 Error/Loss 변화량, Learning Rate를 반영하여 더하여 누적 업데이트함 (Gradient Descent에 의한 Weight 업데이트)

				self.w_[0] += update
				# Perceptron의 입력 Bias (이전 Neuron의 Output에 더해져서 전달되는 양)는 미분하면 0이기에 Weight에 따른 Error/Loss 변화량과 Learning Rate만 반영하여 더하여 누적 업데이트함 (Gradient Descent에 의한 Bias 업데이트)

			errors += int(update != 0.0)	# Error값이 0이 아닌 경우 (Output과 Prediction이 일치하지 않는 경우)에만 Error값을 누적함


		self.errors_.append(errors)

		return self




