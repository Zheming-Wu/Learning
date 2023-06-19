# gpu训练三处加cuda() 分别是：网络模型、损失函数、数据

import torch.optim.optimizer
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

# from model import *

train_data = torchvision.datasets.CIFAR10("dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)  # 直接获取数据集的大小
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_dataloader = DataLoader(train_data, 64, shuffle=True)
test_dataloader = DataLoader(test_data, 64, shuffle=True)


class Wu(nn.Module):
    def __init__(self):
        super(Wu, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    pass


wu = Wu()
if torch.cuda.is_available():
    wu = wu.cuda()  # gpu训练1

loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()  # gpu训练2

optimizer = torch.optim.SGD(wu.parameters(), lr=0.01)

total_train_step = 0
total_test_step = 0
epoch = 20

writer = SummaryWriter("logs")

for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))

    wu.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()  # gpu训练3
            targets = targets.cuda()  # gpu训练3
        outputs = wu(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (total_train_step + 1) % 100 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step + 1, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        total_train_step = total_train_step + 1

    wu.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()  # gpu训练3
                targets = targets.cuda()  # gpu训练3
            outputs = wu(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            total_accuracy = total_accuracy + (outputs.argmax(1) == targets).sum()
    print("整体数据集上的Loss：{}".format(loss.item()))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # torch.save(wu, "wu_{}.pth".format(i))  # 保存每一次训练模型
    # torch.save(wu.state_dict(), "wu_{}".format(i))  # 另一种保存方式
    # print("模型已保存！")

writer.close()
