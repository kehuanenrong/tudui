import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# CIFAR 10 数据集,并转换为Tensor数据类型
test_data = torchvision.datasets.CIFAR10("cifar10", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 利用 DataLoader 去配置加载数据的方式
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 利用Tensorboard去可视化数据集
writer = SummaryWriter("10_2_2")

epochs = 2
# 循环2次数据集
for epoch in range(epochs):
    step = 0
    # 循环完整个数据
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step += 1

writer.close()