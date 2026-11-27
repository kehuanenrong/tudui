from PIL import Image
import torch
from torch import nn
import torchvision

# a = torch.tensor(5)
# print(a) # tensor(5)
# print(a.item())  # 5

image_path = "./custom_data/test/0_cats.png"
image = Image.open(image_path)
image = image.convert("RGB")  # 将图片转换为RGB模式
print(image)  

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

image = transform(image)
print(image.shape)  # torch.Size([3, 32, 32])

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("./tudui_4.pth", weights_only=False)
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print("预测的类别为：", torch.argmax(output, dim=1).item())  # 输出预测的类别