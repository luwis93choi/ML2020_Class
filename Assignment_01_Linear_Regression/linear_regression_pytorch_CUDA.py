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
from torch.autograd import Variable

import matplotlib.pyplot as plt

print("--- ML2020 Assignment 01 : Linear Regression ---")

print("CUDA Ready : " + str(torch.cuda.is_available()))
PROCESSOR = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('./train_data.csv')
x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()

model = nn.Linear(in_features=1, out_features=1, bias=True)
model.to(PROCESSOR)
print('[model] : ' + str(model))

print('[model.weight]')
print(model.weight)

print('[model.bias]')
print(model.bias)

# Hyper Parameters
learning_rate = 0.01
train_step = 3000

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

loss_list = []

fig, axes = plt.subplots(1, 2)

for step in range(train_step):

    x = Variable(x.to(PROCESSOR))
    y = Variable(y.to(PROCESSOR))

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
            axes[0].scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
            axes[0].plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'b--')

            axes[1].set_title('MSE Loss Function')
            axes[1].plot(range(len(loss_list)), loss_list, 'b')

            plt.savefig('./linear_regression_result_with_PyTorch_GPU_CUDA.png')
            plt.draw()

    elif step % 20 == 0:

        print('MSE Loss : ' + str(loss.data.item()))

        plt.suptitle('Linear Regression using PyTorch and GPU/CUDA')

        axes[0].set_title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.item(), model.bias.item()))
        axes[0].set_xlim(0, 11)
        axes[0].set_ylim(0, 8)
        axes[0].scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
        axes[0].plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'b--')
        
        axes[1].set_title('MSE Loss Function')
        axes[1].plot(range(len(loss_list)), loss_list, 'b')
        
        plt.draw()
        plt.pause(0.01)
        axes[0].clear()
