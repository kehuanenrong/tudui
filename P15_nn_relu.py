import torch
from torch import nn
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
import torchvision

input = torch.tensor([[1, -0.5],
                      [-1, 3.0]])
print(input.shape)

output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)

dataset = torchvision.datasets.CIFAR10(root="./cifar10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64,drop_last=True)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        return x

tudui = Tudui()

writer = SummaryWriter("logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step = step)
    output = tudui(imgs)
    writer.add_images("output", output, global_step = step)
    step += 1

writer.close()
