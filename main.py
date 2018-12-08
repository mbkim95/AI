import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from DataSet import *

#  Linear Neural Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_size, 10)
        self.L1 = nn.Linear(10, 6)
        self.L2 = nn.Linear(6, 4)
        self.output_layer = nn.Linear(4, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = self.output_layer(x)
        return x

def init_weights(m):                                        # 초기 Weights 설정
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)

def train(model, input_data, target_data, optimizer, epoch):
    # In training progress
    # 1. turn on training mode
    model.train()
    for batch_idx, (data, target) in enumerate(zip(input_data, target_data)):
        # Input data & matched real numbers
        data, target = data.to(device), target.to(device)
        # 2. Reset optimizer's parameters' gradient
        optimizer.zero_grad()
        # 3. Predict the result(output) with the model
        output = model(input_data)
        target_data = target_data.float()
        output = torch.squeeze(output)
        # 4. Calculate the loss(cost). Loss
        # The loss indicates how different the predicted value is from the actual value
        criterion = nn.MSELoss()
        loss = criterion(output, target_data)
        # 5. Backpropagation & optimization
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(input_data),
                    100. * batch_idx / len(input_data), loss.item()))

def test(model, input_train, input_target):
    model.eval()
    correct = 0
    for data, target in zip(input_train, input_target):
        output_data = model(data)
        index = output_data.long()
        if index == target:
            correct += 1
    percentage = round(100 * (correct / len(input_target)))
    print('Percentage:{}%({}/{})'.format(percentage, correct, len(input_target)))

def divide_Dataset(input, target, device, batch_size, i):
    train_values, test_values = [], []
    for idx, (x_test, y_test) in range(i, i+batch_size) and enumerate(zip(input, target)):
        x_train = torch.Tensor([], device=device)
        y_train = torch.LongTensor([], device=device)
        for train_idx, (x, y) in enumerate(zip(input, target)):
            if train_idx != idx:
                x_train = torch.cat((x_train, x), 0)  # x_train과 x을 이어붙임
                y_train = torch.cat((y_train, y), 0)  # y_train과 y를 이어붙임
                return x_train, y_train

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. Model representation
    input_data, target_data, col = preprocess('./student/student-mat.csv')
    ratio = 10

    # if you want cross validation
    input, target = cross_validation(ratio, input_data, target_data)

    ##
    model = NeuralNet(col, 1).to(device)
    model.apply(init_weights)
    # 2. Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    epoch_range, batch = 50, 5

    # #
    # for epoch in range(0, epoch_range):
    #     # 3. Train the model
    #     train(model, input_data, target_data, optimizer, epoch)
    #     # 4. Evaluate the training model
    #     test(model, input_data, target_data)

    #
    train_values, test_values = [], []
    for idx, (x_test, y_test) in enumerate(zip(input, target)):
        x_train = torch.Tensor([], device=device)
        y_train = torch.LongTensor([], device=device)
        for train_idx, (x, y) in enumerate(zip(input, target)):
            if train_idx != idx:
                x_train = torch.cat((x_train, x), 0)  # x_train과 x을 이어붙임
                y_train = torch.cat((y_train, y), 0)  # y_train과 y를 이어붙임

                train(model, x_train, y_train, optimizer, idx)

                test(model, x_test, y_test)


if __name__ == '__main__':
    main()
