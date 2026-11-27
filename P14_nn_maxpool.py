from requests.packages import target
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./cifar10", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)  # 定义一个最大池化层，池化核大小为3x3，使用ceil_mode=False表示不使用向上取整

    def forward(self, x):
        x = self.maxpool(x)
        return x

tudui = Tudui()

writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = tudui(imgs)
    writer.add_images("output", output, global_step=step)
    step = step + 1

writer.close()