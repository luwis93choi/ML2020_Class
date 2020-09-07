# Reference 01 (Anaconda Installation) : https://docs.anaconda.com/anaconda/install/linux/
# Reference 02 (Anaconda Package Control) : https://niceman.tistory.com/86
# Reference 03 (Anaconda Pytorch Setup) : https://pytorch.org/get-started/locally/
# Reference 04 (Pytorch Linear Regression) : https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/
# Reference 05 (Google Machine Learning Crash Course - Linear Regression) : https://developers.google.com/machine-learning/crash-course/descending-into-ml/video-lecture
# Reference 06 : 인공지능을 위한 수학 (이시카와 아키히코) - 5장 선형회귀
# Reference 07 (Linear Regression From Scratch without any Library) : https://www.kaggle.com/siddhrath/linear-regression-from-scratch-without-any-library

import pandas as pd

import torch
from torch import nn

import matplotlib.pyplot as plt

print("--- ML2020 Assignment 01 : Linear Regression ---")

# Load Dataset
data = pd.read_csv('./train_data.csv')

# Hyper Parameters
learning_rate = 0.01
train_step = 3000

def linear_regression_Pytorch(data, _lr, train_step):

    x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
    y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()

    model = nn.Linear(in_features=1, out_features=1, bias=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=_lr)

    loss_list = []

    fig, axes = plt.subplots(1, 2)
    
    for step in range(train_step):

        prediction = model(x)
        loss = criterion(input=prediction, target=y)

        loss_list.append(loss.data.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == train_step-1:

            plt.suptitle('Linear Regression using PyTorch')

            axes[0].set_title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.item(), model.bias.item()))
            axes[0].set_xlim(0, 11)
            axes[0].set_ylim(0, 8)
            axes[0].scatter(x.data.numpy(), y.data.numpy())
            axes[0].plot(x.data.numpy(), prediction.data.numpy(), 'b--')

            axes[1].set_title('MSE Loss Function')
            axes[1].plot(range(len(loss_list)), loss_list, 'b')

            plt.savefig('./linear_regression_result_with_PyTorch.png')
            plt.draw()

        elif step % 20 == 0:

            print('MSE Loss : ' + str(loss.data.item()))
            
            plt.suptitle('Linear Regression using Pytorch')
            
            axes[0].set_title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.item(), model.bias.item()))
            axes[0].set_xlim(0, 11)
            axes[0].set_ylim(0, 8)
            axes[0].scatter(x.data.numpy(), y.data.numpy())
            axes[0].plot(x.data.numpy(), prediction.data.numpy(), 'b--')

            axes[1].set_title('MSE Loss Function')
            axes[1].plot(range(len(loss_list)), loss_list, 'b')

            plt.draw()
            plt.pause(0.01)
            axes[0].clear()

def predict(_x, _w, _b):

    return _w * _x + _b

def L2_loss_function(_x, _y, _w, _b):

    data_num = len(_x)
    total_error = 0.0

    for i in range(data_num):

        total_error += (_y[i] - (_w * _x[i] + _b))**2

    return total_error / data_num

def update_gradient_descent(_x, _y, _w, _b, learning_rate):

    weight_derivative = 0
    bias_derivative = 0
    data_num = len(_x)

    for i in range(data_num):
        weight_derivative += -2 * _x[i] * (_y[i] - (_x[i] * _w + _b))
        bias_derivative += -2 * (_y[i] - (_x[i] * _w + _b))

    _w -= (weight_derivative / data_num) * learning_rate
    _b -= (bias_derivative / data_num) * learning_rate

    return _w, _b

def linear_regression_NoPytorch(data, init_w, init_b, learning_rate, train_step):

    x = data['x'].values
    y = data['y'].values

    loss_list = []

    fig, axes = plt.subplots(1, 2)

    for step in range(train_step):

        init_w, init_b = update_gradient_descent(x, y, init_w, init_b, learning_rate)

        loss = L2_loss_function(x, y, init_w, init_b)
        loss_list.append(loss)

        if step == train_step-1:

            plt.suptitle('Linear Regression without PyTorch')

            axes[0].set_title('loss={:.4}, w={:.4}, b={:.4}'.format(loss, init_w, init_b))
            axes[0].set_xlim(0, 11)
            axes[0].set_ylim(0, 8)
            axes[0].scatter(x, y)
            axes[0].plot(x, predict(x, init_w, init_b), 'b--')

            axes[1].set_title('MSE Loss Function')
            axes[1].plot(range(len(loss_list)), loss_list, 'b')

            plt.savefig('./linear_regression_result_without_PyTorch.png')
            plt.show()

        elif step % 20 == 0:

            print('MSE Loss : ' + str(loss))
            
            plt.suptitle('Linear Regression without Pytorch')

            axes[0].set_title('loss={:.4}, w={:.4}, b={:.4}'.format(loss, init_w, init_b))
            axes[0].set_xlim(0, 11)
            axes[0].set_ylim(0, 8)
            axes[0].scatter(x, y)
            axes[0].plot(x, predict(x, init_w, init_b), 'b--')
            
            axes[1].set_title('MSE Loss Function')
            axes[1].plot(range(len(loss_list)), loss_list, 'b')
            
            plt.draw()
            plt.pause(0.01)
            axes[0].clear()

    return init_w, init_b

linear_regression_Pytorch(data, learning_rate, train_step)

linear_regression_NoPytorch(data, 0, 0, learning_rate, train_step)