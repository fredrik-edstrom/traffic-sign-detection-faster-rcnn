"""
This file contains our initial neural network model.
We used it with the wikipedia dataset
"""

from torch.nn import BatchNorm2d, Conv2d, CrossEntropyLoss, Linear, MaxPool2d, Module, ReLU, Sequential
from torch.optim import Adam


class ConvolutionalNeuralNetwork(Module):
    def __init__(self, size: int) -> None:
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = Sequential(
            Linear(2500, size)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x

    def train_nn(self, epoch: int, optimizer: Adam, criterion: CrossEntropyLoss, train_x, train_y, test_x, test_y) -> None:
        import torch

        train_losses, val_losses = [], []

        if torch.cuda.is_available():
            model = self.cuda()
            criterion = criterion.cuda()

        self.train()
        tr_loss = 0

        if torch.cuda.is_available():
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            test_x = test_x.cuda()
            test_y = test_y.cuda()

        optimizer.zero_grad()

        output_train = self(train_x)
        output_val = self(test_x)

        loss_train = criterion(output_train, train_y)
        loss_val = criterion(output_val, test_y)
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()

        if epoch % 2 == 0:
            print(f'Epoch: {epoch + 1}\tLoss: {loss_val}')
