import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./cifar10", train=True, download=True, transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root="./cifar10", train=False, download=True, transform=dataset_transform)

# print(test_set[0])
# print(test_set.classes)

# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()  # 显示图片

writer = SummaryWriter("P9_logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
    print(test_set.classes[target])

writer.close()