import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):

    def __init(self):
        xy = np.loadtxt('./data/student-por.csv', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0: -1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = torch.nn.Linear(8, 4)
        self.l2 = torch.nn.Linear(4, 6)
        self.l3 = torch.nn.Linear(6, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

model = MyModel()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        y_pred = model(input)

        loss = criterion(y_pred, labels)
        print(epoch, i, loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
