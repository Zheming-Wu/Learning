import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)

dataloader = DataLoader(dataset, 64, True, drop_last=False)

class Wu(nn.Module):

    def __init__(self):
        super(Wu, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, 1, 2)
        # self.maxpool1 = MaxPool2d(2, ceil_mode=True)  # celi_model用于设置余数要不要算进来计算，True是要的意思
        # self.conv2 = Conv2d(32, 32, 5, 1, 2)
        # self.maxpool2 = MaxPool2d(2, ceil_mode=True)
        # self.conv3 = Conv2d(32, 64, 5, 1, 2)
        # self.maxpool3 = MaxPool2d(2, ceil_mode=True)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2, ceil_mode=True),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2, ceil_mode=True),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2, ceil_mode=True),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)

        x = self.model1(x)

        return x


wu = Wu()
loss = nn.CrossEntropyLoss()

for data in dataloader:
    imgs, targets = data
    # print(imgs.shape, targets.shape)
    output = wu(imgs)
    # print(output)
    result_loss = loss(output, targets)
    # print(output.shape)
    print(result_loss)
    result_loss.backward()
    print()
