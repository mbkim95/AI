import time
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np

from DataSet import *


#  Linear Neural Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_size, 18, bias=False)
        self.L1 = nn.Linear(18, 12, bias=False)
        self.output_layer = nn.Linear(12, output_size, bias=False)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.L1(x))
        x = self.output_layer(x)
        return x

start_time = time.time()                                                                # 시작 시간 측정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                   # cuda를 쓸지 cpu를 쓸지 세팅
epoch, batch_size = 10000, 20                                                           # 학습 epoch와 batch_size 설정

input_matrix, target_matrix, features_counts = preprocess('./student/student-mat.csv')                             # 데이터 전처리,              feature_counts -> 특성 개수
ratio = 10                                                                              # 교차검증 비율 설정
input_arrays, target_arrays = cross_validation(ratio, input_matrix, target_matrix)      # 교차검증하도록 세팅
# criterion = nn.CrossEntropyLoss()                                                       # cost함수 설정
criterion = nn.MSELoss()

#  Train Model
def train(epoch, input_model, input_training, input_target):
    length = len(input_training)
    quo = length // batch_size                  # 소수부부은 버려서 나눗셈
    for epoch_idx in range(epoch):
        for idx in range(quo):
            start = idx * batch_size
            end = (idx + 1) * batch_size if idx != quo - 1 else length - 1
            mini_training = input_training[start:end]
            mini_target = input_target[start:end]
            output_data = input_model(mini_training)
            mini_target = mini_target.float()
            # mini_target = mini_target.long()
            output_data = output_data.float()
            output_data = torch.squeeze(output_data)
            loss = criterion(output_data, mini_target)
            # model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch_idx+1) % 50 == 0:
            print('Train Epoch: {}/{}\tLoss: {:.6f}'.format(epoch_idx+1, epoch, loss.data[0]))
    return loss


#  Test Model
def test(text, input_model, input_train, input_target):
    correct = 0
    for data, target in zip(input_train, input_target):
        output_data = input_model(data)
        index = output_data.max(0)[1]
        if index == target:
            correct += 1
    percentage = round(100 * (correct / len(input_target)))
    print('{} Percentage:{}%({}/{})'.format(text, percentage, correct, len(input_target)))
    return percentage


#  Init Weights using Xavior Function
def init_weights(m):                                        # 초기 Weights 설정
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)


#  Divide DataSets
train_values, test_values = [], []
for idx, (x_test, y_test) in enumerate(zip(input_arrays, target_arrays)):
    x_train = torch.Tensor([], device=device)
    y_train = torch.LongTensor([], device=device)
    for train_idx, (x, y) in enumerate(zip(input_arrays, target_arrays)):
        if train_idx != idx:
            x_train = torch.cat((x_train, x), 0)                    # x_train과 x을 이어붙임
            y_train = torch.cat((y_train, y), 0)                    # y_train과 y를 이어붙임

    model = NeuralNet(features_counts, 1)
    # model = BasicCNN(len(x_train), 4, features_counts)
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.03, weight_decay=0.04)                       # optimizer 설정
    loss = train(epoch, model, x_train, y_train)                                                # 학습과정
    print('R{} Train Loss: {}'.format(idx+1, loss.data[0]))
    train_values.append(test('R{} Train'.format(idx+1), model, x_train, y_train))
    test_values.append(test('R{} Test'.format(idx+1), model, x_test, y_test))
    print()

print("time: {}".format(time.time()-start_time))
overall_train = round(sum(train_values) / ratio)
overall_test = round(sum(test_values) / ratio)
for test_value in test_values:
    print("{} ".format(test_value), end="")
print()
print("overall train's percentage: {}%, test's percentage: {}%".format(overall_train, overall_test))
