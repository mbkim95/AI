import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from DataSet import *

#  Linear Neural Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_size, 120)
        self.L1 = nn.Linear(120, 100)
        self.output_layer = nn.Linear(100, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.L1(x))
        x = self.output_layer(x)
        return x

# def init_weights(m):                                        # 초기 Weights 설정
#     if isinstance(m, nn.Linear):
#         size = m.weight.size()
#         fan_out = size[0]  # number of rows
#         fan_in = size[1]  # number of columns
#         variance = np.sqrt(2.0 / (fan_in + fan_out))
#         m.weight.data.normal_(0.0, variance)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

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
        output = model(data)
        target = target.float()
        output = torch.squeeze(output)
        # 4. Calculate the loss(cost). Loss
        # The loss indicates how different the predicted value is from the actual value
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        # 5. Backpropagation & optimization
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(input_data),
                    100. * batch_idx / len(input_data), loss.item()))

def test(model, input_test, input_target):
    model.eval()
    # correct = 0
    for data, target in zip(input_test, input_target):
        output_data = model(data)
        target = target.float()
        l = output_data
        value = output_data
        # value = torch.round(output_data).long()
        # value = value.long()
        criterion = nn.MSELoss()
        loss = criterion(l, target)
        correct = 1 - abs(value - target)/21
        # if value == t:
        #     correct += 1
    # percentage = round(100 * (correct / len(input_target)))
    print('{} {} {}'.format(value, target, correct))
    percentage = correct * 100
    percentage = torch.squeeze(percentage)
    # print('Percentage:{}%({}/{})'.format(percentage, correct, len(input_target)))
    print('Percentage:{}%'.format(percentage))
    print('Loss:{}'.format(loss))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. Model representation
    train_input, train_target, test_input, test_target, col = preprocess('./student/student-por.csv')

    ##
    model = NeuralNet(col, 1).to(device)
    model.apply(init_weights)
    # 2. Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    epoch_range, batch = 500, 5

    #
    for epoch in range(0, epoch_range):
        # 3. Train the model
        train(model, train_input, train_target, optimizer, epoch)
        # 4. Evaluate the training model
    test(model, test_input, test_target)


if __name__ == '__main__':
    main()
