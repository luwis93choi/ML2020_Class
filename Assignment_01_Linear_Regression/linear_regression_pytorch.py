# Reference 01 (Anaconda Installation) : https://docs.anaconda.com/anaconda/install/linux/
# Reference 02 (Anaconda Package Control) : https://niceman.tistory.com/86
# Reference 03 (Anaconda Pytorch Setup) : https://pytorch.org/get-started/locally/
# Reference 04 (Pytorch Linear Regression) : https://greeksharifa.github.io/pytorch/2018/11/02/pytorch-usage-02-Linear-Regression-Model/
# Reference 05 (Google Machine Learning Crash Course - Linear Regression) : https://developers.google.com/machine-learning/crash-course/descending-into-ml/video-lecture
# Reference 06 : 인공지능을 위한 수학 (이시카와 아키히코)

import pandas as pd

import torch
from torch import nn

import matplotlib.pyplot as plt

print("--- ML2020 Assignment 01 : Linear Regression ---")

data = pd.read_csv('./train_data.csv')
x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()

model = nn.Linear(in_features=1, out_features=1, bias=True)
print('[model] : ' + str(model))

print('[model.weight]')
print(model.weight)

print('[model.bias]')
print(model.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

loss_list = []
for step in range(3000):

    prediction = model(x)
    loss = criterion(input=prediction, target=y)

    loss_list.append(loss.data.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:

        print('MSE Loss : ' + str(loss.data.item()))

        plt.subplot(1, 2, 1)
        plt.title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.item(), model.bias.item()))
        plt.xlim(0, 11)
        plt.ylim(0, 8)
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'b--')
        plt.draw()

        plt.subplot(1, 2, 2)
        plt.title('MSE Loss Function')
        plt.plot(range(len(loss_list)), loss_list, 'b')
        plt.pause(0.01)

        plt.subplot(1, 2, 1)
        plt.clf()

plt.subplot(1, 2, 1)
plt.title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.item(), model.bias.item()))
plt.xlim(0, 11)
plt.ylim(0, 8)
plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
plt.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'b--')

plt.subplot(1, 2, 2)
plt.title('MSE Loss Function')
plt.plot(range(len(loss_list)), loss_list, 'b')
plt.savefig('./linear_regression_result.png')
plt.show()