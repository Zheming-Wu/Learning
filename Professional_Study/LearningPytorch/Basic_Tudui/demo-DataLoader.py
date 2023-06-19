import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# test data set
test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=transforms.ToTensor())

# test loader
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中的第一张图片
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1
        pass
    pass

writer.close()

