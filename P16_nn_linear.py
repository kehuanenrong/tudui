import torch
from torch import nn
from torch.nn import Linear
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./cifar10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear = Linear(64*3*32*32, 10)

    def forward(self, x):
        x = self.linear(x)
        return x

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    print(output.shape)
    output_t = torch.flatten(imgs)
    print(output_t.shape)
    output = tudui(output_t)
    print(output.shape)