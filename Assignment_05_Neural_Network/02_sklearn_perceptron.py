####################################################################################
### Q1-2 : Iris Classification using scikit learn Perceptron and standardization ###
####################################################################################

from sklearn import datasets
import numpy as np

### 데이터셋 준비 ###
iris = datasets.load_iris()	# Iris 데이터셋 로딩

X = iris.data[:, [2, 3]]	# Feature 데이터로 데이터셋의 2번 Column (Petal Length), 3번 Column (Petal Width) 선정
y = iris.target			# 데이터셋의 Label/Target 준비

print('class label :', np.unique(y))	# Label/Target의 종류 확인 (np.unique를 사용해서 Label 종류만 선택적으로 출력함)

from sklearn.model_selection import train_test_split

# Train 70%, Test 30%로 데이터셋을 분리함
# 데이터셋을 매번 분리할 때마다 Reproducibility를 보장하기 위해 random_state를 고정함
# startify는 데이터셋 분리 시 분리된 각 데이터셋 내부에 Label별 데이터가 골고루 분포될 수 있도록 보장하는 방법임
# stratify=y를 설정하여 Label별 데이터가 Train, Test에 동일하게 골고루 분포되게 만듬
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# 각 데이터셋 종류별로 Label 데이터가 동일하게 골고루 분포되게 분리가 되었는지 확인함
print('label count of y:', np.bincount(y))
print('label count of y_train:', np.bincount(y_train))
print('label count of y_test:', np.bincount(y_test))

### 데이터셋 정규화 ###
# 데이터셋 Standardization을 위해 z-score로 정규화하는 StandardScaler 사용
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()			# Standard Scaler 객체 선언
sc.fit(X_train)				# Feature 데이터에 대한 Mean과 Standard Deviation 산출함
X_train_std = sc.transform(X_train)	# Train 데이터셋 정규화
X_test_std = sc.transform(X_test)	# Test 데이터셋 정규화

### Perceptron 학습 수행 ###
# sklearn Perceptron 사용
from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter=40, eta0=0.1, tol=1e-3, random_state=1)
# 반복 횟수 40, Learning Rate 0.1, 정지 기준 Tolerance 0.001, 최초 weight 초기화 개수 (Seed, random_state) 1개로 Perceptron 선언

# 정규화된 Train 데이터셋에 대해 Perceptron 학습 수행
ppn.fit(X_train_std, y_train)

# 정규화된 Test 데이터셋에 대해 Prediction 수행
y_pred = ppn.predict(X_test_std)
print('The number of misclassified sample : %d' % (y_test != y_pred).sum())	# Prediction 틀린 개수 출력

from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))	# Test 데이터셋과 Prediction을 비교하여 정확도 출력

### Perceptron 훈련 결과 Decision Boundary로 표현 ###
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

	# 그래프 상 각 데이터의 Label과 영역을 표시하기 위한 Marker와 Colormap 선언
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])	# 각 Label별로 리스트 상 순서에 따라 고유의 색이 사용되게함

	# Decision Boundary 생성
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1	# Dataset의 0번 Column 데이터 범위
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1	# Dataset의 1번 Column 데이터 범위

	# np.meshgrid를 사용하여 Dataset의 0번, 1번 데이터 범위로 구성된 2D Matrix 생성
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
		   np.arange(x2_min, x2_max, resolution))

	# Dataset의 0번, 1번 데이터를 각각 Column-wise로 합쳐서 Classifier에게 전달하여 Prediction 수행함
	# Dataset 전 범위에 걸쳐서 Prediction을 수행함
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)	# Prediction 결과를 입력 데이터 형태와 동일하게 재구성함

	plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
	# 데이터 전 범위에 대한 Prediction 결과에서 0인 경우 빨간색, 1인 경우 파란색, 2인 경우 초록색으로 칠해서 Decision Boundary에 의한 영역분할을 나타냄
	plt.xlim(xx1.min(), xx1.max())	# 그래프 X 범위 설정
	plt.ylim(xx2.min(), xx2.max())	# 그래프 y 범위 설정

	# 주어진 Iris 데이터셋에 대한 Prediction 결과를 Scatter Plot으로 추가로 그려넣음
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], 
			    y=X[y == cl, 1],
			    alpha=0.8, 
			    c=colors[idx],
			    marker=markers[idx], 
			    label=cl, 
			    edgecolor='black')

	# Test 데이터셋이 어떤 지점에 있는지 Scatter Plot으로 추가로 그려넣음
	if test_idx:
		X_test, y_test = X[test_idx, :], y[test_idx]

		plt.scatter(X_test[:, 0],
			    X_test[:, 1],
			    facecolors='none',
			    edgecolor='black',
			    alpha=1.0,
			    linewidth=1,
			    marker='o',
			    s=100, 
			    label='test set')

# 그래프로 그리기 위해 정규화된 모든 데이터셋을 합침
X_combined_std = np.vstack((X_train_std, X_test_std))	# 정규화된 Feature 데이터셋을 합침
y_combined = np.hstack((y_train, y_test))		# 정규화된 Label 데이터셋을 합침

# Perceptron의 Prediction 결과를 Decision Boundary로 나타냄
# Iris 데이터셋 모든 Label에 대해서 수행함
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')	# 그래프 X축 설명
plt.ylabel('petal width [standardized]')	# 그래프 y축 설명
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()



