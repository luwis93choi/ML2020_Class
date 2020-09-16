# Reference 01 (Anaconda Installation) : https://docs.anaconda.com/anaconda/install/linux/
# Reference 02 (Anaconda Package Control) : https://niceman.tistory.com/86
# Reference 03 (Anaconda Pytorch Setup) : https://pytorch.org/get-started/locally/
# Reference 04 (Pytorch Linear Regression) : https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/
# Reference 05 (Google Machine Learning Crash Course - Linear Regression) : https://developers.google.com/machine-learning/crash-course/descending-into-ml/video-lecture
# Reference 06 : 인공지능을 위한 수학 (이시카와 아키히코) - 5장 선형회귀
# Reference 07 (Linear Regression From Scratch without any Library) : https://www.kaggle.com/siddhrath/linear-regression-from-scratch-without-any-library

import pandas as pd                 # pandas (Python Data Analysis Library) - 데이터셋을 읽고 분석하기 위해 사용 pandas 라이브러리 사용

import torch                        # torch (Pytorch) - Machine Learning 및 Neural Network 라이브러리 사용
from torch import nn                # Pytorch의 Neural Network 라이브러리 사용

import matplotlib.pyplot as plt     # matplotlib - 그래프 작성 및 출력을 위한 라이브러리 사용

print("--- ML2020 Assignment 01 : Linear Regression ---")

# pandas를 이용하여 csv 파일로 저장된 데이터셋 읽음
data = pd.read_csv('./train_data.csv')

# Hyper Parameters - Machine Learning의 Training 및 Inference 전반적으로 영향을 주는 Hyper Parameter 변수 선언
learning_rate = 0.01    # 학습 과정시 Loss Function 최소값를 도달하기 위해 매번마다 Weight와 Bias에 얼마나 크게 변화를 줄 지 결정 / Loss Function 최소값 탐색의 한 단계의 크기 결정
train_step = 3000       # 전체 학습의 횟수 결정

# PyTorch를 사용한 선형 회귀 (Linear Regression) 함수
def linear_regression_Pytorch(data, _lr, train_step):

    x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()     # csv에서 읽은 x축 데이터를 numpy 배열에서 tensor 형태로 변환
    y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()     # csv에서 읽은 y축 데이터를 numpy 배열에서 tensor 형태로 변환

    model = nn.Linear(in_features=1, out_features=1, bias=True)     #  Linear Regression 모델 선언
                                                                    #  - in_features : Linear Regression 모델의 입력 데이터의 개수 (x 1개) (weight가 아닌 모델에 입력되는 데이터)
                                                                    #  - out_features : Linear Regression 모델의 출력 데이터의 개수 (y 1개) (weight가 아닌 모델이 출력하는 데이터)
                                                                    #  *** 학습의 결과가 ML/NN 모델이며, 모델은 입력 데이터를 받아 특정 데이터를 출력되게 만들어짐 ***

    criterion = nn.MSELoss()                                        # MSE (Mean Squared Error)를 Loss Function으로 사용
    optimizer = torch.optim.Adam(params=model.parameters(), lr=_lr) # Loss Function 최소값 탐색을 위해 사용할 Optimizer로 Adam Optimizer 사용하며, 탐색의 한 단계의 크기 Learning Rate를 설정함

    loss_list = []  # 추후 Loss Function 그래프를 그리기 위해 Loss 값을 저장할 리스트

    fig, axes = plt.subplots(1, 2)  # Linear Regression 결과를 1x2 그래프 2개로 그릴 예정
    
    # 상기 정의한 train_step 값만큼 전체 데이터셋을 가지고 학습을 반복함
    for step in range(train_step):

        prediction = model(x)                           # 현재 모델 / 현재까지 학습된 모델을 가지고 입력 데이터 x에 대한 출력 데이터를 산출함
        loss = criterion(input=prediction, target=y)    # 현재 모델 / 현재까지 학습된 모델이 출력한 데이터가 목표하는 원래 y값에 비교해서 얼마나 Loss가 발생하는지 MSE Loss Function값으로 구함

        loss_list.append(loss.data.item())  # 현재 모델의 Loss값을 저장함

        optimizer.zero_grad()   # Optimizer의 grad값을 0으로 설정함 / PyTorch는 Parameter들이 gradient를 계산해줄때 grad가 누적됨. 이에 따라 Gradient를 다시 계산할 때는 0으로 초기화해야함
        loss.backward()         # Gradient 계산에 따른 Backpropagation(역전파) 수행
        optimizer.step()        # 계산한 Gradient값을 기반으로 Parameter (Weight, Bias)를 Learning Rate에 비례해서 업데이트해줌

        # 학습의 마지막 단계에 도달했을 시 학습 결과에 대한 그래프를 저장함
        if step == train_step-1:

            plt.suptitle('Linear Regression using PyTorch') # 메인 Title 작성

            # 좌측 그래프는 현재 학습된 결과(Weight, Bias)에 기반한 Linear 그래프를 데이터셋 분포도 상에 그림
            axes[0].set_title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.item(), model.bias.item()))  # 좌측 그래프 제목에 현재 학습 모델의 Loss, Weight, Bias를 보여줌
            axes[0].set_xlim(0, 11)                                         # 그래프상 x축의 범위 설정
            axes[0].set_ylim(0, 8)                                          # 그래프상 y축의 범위 설정
            axes[0].scatter(x.data.numpy(), y.data.numpy())                 # 데이터셋 분포도 그래프 그림
            axes[0].plot(x.data.numpy(), prediction.data.numpy(), 'b--')    # 데이터셋 분포도 상에 현재 학습된 결과(Weight, Bias)에 기반한 Linear 그래프를 그림

            # 우측 그래프는 학습 과정에서 산출된 Loss의 기록을 그림
            axes[1].set_title('MSE Loss Function')                          # 우측 그래프의 제목 설정
            axes[1].plot(range(len(loss_list)), loss_list, 'b')             # 우측 그래프에 현재까지의 Loss값 기록을 그래프로 그림

            plt.savefig('./linear_regression_result_with_PyTorch.png')      # 최종 결과 그래프 저장
            plt.draw()

        # 20번마다 Loss 값을 그래프에 기록하고, 학습결과 (Weight, Bias)에 따라 데이터셋 분포도에 Linear 그래프를 그림
        elif step % 20 == 0:

            print('MSE Loss : ' + str(loss.data.item()))    # 현재 Loss값을 CLI (Command Line Interface) 콘솔창에 보여줌
            
            plt.suptitle('Linear Regression using Pytorch')
            
            # 좌측 그래프는 현재 학습된 결과에 기반한 Linear 그래프를 데이터셋 분포도상에 그림
            axes[0].set_title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.item(), model.bias.item()))  # 우측 그래프 제목에 현재 학습 모델의 Loss, Weight, Bias를 보여줌
            axes[0].set_xlim(0, 11)                                         # 그래프상 x축의 범위 설정
            axes[0].set_ylim(0, 8)                                          # 그래프상 y축의 범위 설정
            axes[0].scatter(x.data.numpy(), y.data.numpy())                 # 데이터셋 분포도 그래프 그림
            axes[0].plot(x.data.numpy(), prediction.data.numpy(), 'b--')    # 데이터셋 분포도 상에 현재 학습된 결과(Weight, Bias)에 기반한 Linear 그래프를 그림
            
            # 우측 그래프는 학습 과정에서 산출된 Loss의 기록을 그림
            axes[1].set_title('MSE Loss Function')                          # 우측 그래프의 제목 설정
            axes[1].plot(range(len(loss_list)), loss_list, 'b')             # 우측 그래프에 현재까지의 Loss값 기록을 그래프로 그림

            plt.draw()          # 좌측/우측 그래프 렌더링 수행
            plt.pause(0.01)     # 그래프 작성 딜레이
            axes[0].clear()     # 좌측 그래프 업데이트를 위한 화면 Clear

# 주어진 Weight와 Bias를 기반으로 입력 데이터 x에 대한 결과값 출력함수
def predict(_x, _w, _b):

    return _w * _x + _b     # Linear 함수 : y = w * x + b

# 주어진 Weight와 Bias를 기반으로 입력 데이터 x가 출력한 값과 출력 데이터 y 간의 MSE를 계산하여 Loss를 산출하는 함수
def MSE_loss_function(_x, _y, _w, _b):

    data_num = len(_x)      # 전체 데이터 개수
    total_error = 0.0       # Squared Error 누적값

    # 전체 x와 y 데이터에 대해서 Squared Error 누적값을 구함
    for i in range(data_num):

        total_error += (_y[i] - (_w * _x[i] + _b))**2   # 모델이 현재까지 학습한 Weight와 Bias값을 기반으로 
                                                        # i번째 x 데이터의 출력 결과(_w * _x[i] + _b)와 i번째 y 데이터(y[i]) 간의 Error값을 구하고 제곱함 
                                                        # Error값을 누적하여 Squared Error 누적값을 산출함

    return total_error / data_num   # 누적된 Error값에 대해서 평균을 내어 MSE값을 출력값으로 제공함

# Gradient Descent를 이용하여 Loss 최소값에 도달하는 함수
def update_gradient_descent(_x, _y, _w, _b, learning_rate):

    # MSE Loss Function은 학습의 대상인 Weight와 Bias에 대해서 2차함수임
    # Loss가 최소값에 곳은 MSE Loss의 변화량/미분값이 0이 되는 순간임
    # MSE Loss는 2개의 변수에 의해서 제어되기 때문에 미분을 할 경우 Weight와 Bias에 대해서 각각 편미분을 해줘야함
    # 
    # Loss Gradient = (d Loss/d weight, d Loss/d bias) 

    weight_derivative = 0       # Weight값에 대한 편미분 결과값
    bias_derivative = 0         # Bias값에 대한 편미분 결과값
    data_num = len(_x)          # 전체 데이터 개수

    # Gradient Descent는 전체 학습 데이터에 대해서 Loss Gradient를 모두 합해서 최소값에 도달하는지 판단함
    for i in range(data_num):
        weight_derivative += -2 * _x[i] * (_y[i] - (_x[i] * _w + _b))   # MSE Loss 공식을 Weight에 대해서 편미분한 결과에 각각 현재 Weight, 현재 Bias, x, y값을 넣어서 누적함
        bias_derivative += -2 * (_y[i] - (_x[i] * _w + _b))             # MSE Loss 공식을 Bias에 대해서 편미분한 결과에 각각 현재 Weight, 현재 Bias, x, y값을 넣어서 누적함

    _w -= (weight_derivative / data_num) * learning_rate    # 새롭게 업데이트 되는 Weight값은 누적된 편미분값(누적 변화량)을 평균내어 평균 변화량에 Learning Rate를 반영하여 차감함 
                                                            # (최소값을 향해서 가는 Negative Gradient Descent)
                                                            # Learning Rate에 비례해서 Weight가 한 단계 넘어갈 수 있게함

    _b -= (bias_derivative / data_num) * learning_rate      # 새롭게 업데이트 되는 Bias값은 누적된 편미분값(누적 변화량)을 평균내어 평균 변화량에 Learning Rate를 반영하여 차감함 
                                                            # (최소값을 향해서 가는 Negative Gradient Descent)
                                                            # Learning Rate에 비례해서 Bias가 한 단계 넘어갈 수 있게함

    return _w, _b   # 매 학습은 Gradient Descent를 통해 최소값을 한단계 씩 도달하는 과정이며, 이에 따라 업데이트된 Weight, Bias값을 출력해줌

# PyTorch를 사용하지 않고 MSE Loss Function 공식과 Gradient Descent의 편미분 공식을 기반으로한 선형 회귀 (Linear Regression) 함수
def linear_regression_NoPytorch(data, init_w, init_b, learning_rate, train_step):

    x = data['x'].values    # 'x' 라벨의 입력 데이터를 읽어옴
    y = data['y'].values    # 'y' 라벨의 출력 데이터를 읽어옴

    loss_list = []          # 추후 Loss Function 그래프를 그리기 위해 Loss 값을 저장할 리스트

    fig, axes = plt.subplots(1, 2)  # Linear Regression 결과를 1x2 그래프 2개로 그릴 예정
    
    # 상기 정의한 train_step 값만큼 전체 데이터셋을 가지고 학습을 반복함
    for step in range(train_step):

        # 매단계마다 현재 모델의 Weight, Bias를 기반으로 Loss를 계산하고, 
        # Negative Gradient Descent를 통해 Weight와 Bias값이 Gradient가 최소가 되는 방향으로 업데이트 되게 만듬
        init_w, init_b = update_gradient_descent(x, y, init_w, init_b, learning_rate)

        loss = MSE_loss_function(x, y, init_w, init_b)    # 현재 모델 / 현재까지 학습된 모델이 출력한 데이터가 목표하는 원래 y값에 비교해서 얼마나 Loss가 발생하는지 MSE Loss Function값으로 구함
        loss_list.append(loss)                            # 현재 모델의 Loss값을 저장함
        
        # 학습의 마지막 단계에 도달했을 시 학습 결과에 대한 그래프를 저장함
        if step == train_step-1:

            plt.suptitle('Linear Regression without PyTorch')   # 메인 Title 작성

            # 좌측 그래프는 현재 학습된 결과(Weight, Bias)에 기반한 Linear 그래프를 데이터셋 분포도 상에 그림
            axes[0].set_title('loss={:.4}, w={:.4}, b={:.4}'.format(loss, init_w, init_b))  # 좌측 그래프 제목에 현재 학습 모델의 Loss, Weight, Bias를 보여줌
            axes[0].set_xlim(0, 11)                                         # 그래프상 x축의 범위 설정
            axes[0].set_ylim(0, 8)                                          # 그래프상 y축의 범위 설정
            axes[0].scatter(x, y)                                           # 데이터셋 분포도 그래프 그림
            axes[0].plot(x, predict(x, init_w, init_b), 'b--')              # 데이터셋 분포도 상에 현재 학습된 결과(Weight, Bias)에 기반한 Linear 그래프를 그림

            # 우측 그래프는 학습 과정에서 산출된 Loss의 기록을 그림
            axes[1].set_title('MSE Loss Function')                          # 우측 그래프의 제목 설정
            axes[1].plot(range(len(loss_list)), loss_list, 'b')             # 우측 그래프에 현재까지의 Loss값 기록을 그래프로 그림

            plt.savefig('./linear_regression_result_without_PyTorch.png')   # 최종 결과 그래프 저장
            plt.show()
            
        # 20번마다 Loss 값을 그래프에 기록하고, 학습결과 (Weight, Bias)에 따라 데이터셋 분포도에 Linear 그래프를 그림
        elif step % 20 == 0:

            print('MSE Loss : ' + str(loss))    # 현재 Loss값을 CLI (Command Line Interface) 콘솔창에 보여줌
            
            plt.suptitle('Linear Regression without Pytorch')
            
            # 좌측 그래프는 현재 학습된 결과에 기반한 Linear 그래프를 데이터셋 분포도상에 그림
            axes[0].set_title('loss={:.4}, w={:.4}, b={:.4}'.format(loss, init_w, init_b))  # 우측 그래프 제목에 현재 학습 모델의 Loss, Weight, Bias를 보여줌
            axes[0].set_xlim(0, 11)                                         # 그래프상 x축의 범위 설정
            axes[0].set_ylim(0, 8)                                          # 그래프상 y축의 범위 설정
            axes[0].scatter(x, y)                                           # 데이터셋 분포도 그래프 그림
            axes[0].plot(x, predict(x, init_w, init_b), 'b--')              # 데이터셋 분포도 상에 현재 학습된 결과(Weight, Bias)에 기반한 Linear 그래프를 그림
            
            # 우측 그래프는 학습 과정에서 산출된 Loss의 기록을 그림
            axes[1].set_title('MSE Loss Function')                          # 우측 그래프의 제목 설정
            axes[1].plot(range(len(loss_list)), loss_list, 'b')             # 우측 그래프에 현재까지의 Loss값 기록을 그래프로 그림
            
            plt.draw()          # 좌측/우측 그래프 렌더링 수행
            plt.pause(0.01)     # 그래프 작성 딜레이
            axes[0].clear()     # 좌측 그래프 업데이트를 위한 화면 Clear

    return init_w, init_b   # 최종 학습 모델의 Weight, Bias 출력

linear_regression_Pytorch(data, learning_rate, train_step)              # PyTorch를 사용하여 Linear Regerssion 수행

linear_regression_NoPytorch(data, 0, 0, learning_rate, train_step)      # PyTorch를 사용하지 않고 MSE Loss Function 공식과 Gradient Descent의 편미분 공식을 기반으로한 Linear Regerssion 수행
