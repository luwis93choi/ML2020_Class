################################################################
### Q1-1 : Iris Classification using scikit learn Perceptron ###
################################################################

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()	# Iris 데이터셋 로딩

X = iris.data[:, [2, 3]]	# Feature 데이터로 데이터셋의 2번 Column (Petal Length), 3번 Column (Petal Width) 선정
y = iris.target			# 데이터셋의 Label/Target 준비

print('class label :', np.unique(y))	# Label/Target의 종류 확인 (np.unique를 사용해서 Label 종류만 선택적으로 출력함)

from sklearn.model_selection import train_test_split

# Train 70%, Test 30%로 데이터셋을 분리함
# 데이터셋을 매번 분리할 때마다 Reproducibility를 보장하기 위해 random_state를 고정함
# startify는 데이터셋 분리 시 분리된 각 데이터셋 내부에 Label별 데이터가 골고루 분포될 수 있도록 보장하는 방법임.
# stratify=y를 설정하여 Label별 데이터가 Train, Test에 동일하게 골고루 분포되게 만듬
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# 각 데이터셋 종류별로 Label 데이터가 동일하게 골고루 분포되게 분리가 되었는지 확인함
print('label count of y:', np.bincount(y))
print('label count of y_train:', np.bincount(y_train))
print('label count of y_test:', np.bincount(y_test))

# 데이터셋 Standardization을 위해 z-score로 정규화하는 StandardScaler 사용
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()			# Standard Scaler 객체 선언
sc.fit(X_train)				# Feature 데이터에 대한 Mean과 Standard Deviation 산출함
X_train_std = sc.transform(X_train)	# Train 데이터셋 정규화
X_test_std = sc.transform(X_test)	# Test 데이터셋 정규화

# sklearn Perceptron 사용
from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter=40, eta0=0.1, tol=1e-3, random_state=1)
# 반복 횟수 40, Learning Rate 0.1, 정지 기준 Tolerance 0.001

ppn.fit(X_train_std, y_train)
