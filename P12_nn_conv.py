import torch
import torch.nn.functional as F

input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
])

kernel = torch.tensor([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])

input = torch.reshape(input, (1, 1, 5, 5))  # 将输入张量调整为4D张量，形状为(1, 1, 5, 5)
kernel = torch.reshape(kernel, (1, 1, 3, 3))  # 将卷积核调整为4D张量，形状为(1, 1, 3, 3)    

print(input.shape)  # 输出: torch.Size([1, 1, 5, 5])
print(kernel.shape)  # 输出: torch.Size([1, 1, 3, 3])

output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)  # 使用步长为2进行卷积操作
print(output2)  # 输出卷积结果

output3 = F.conv2d(input, kernel, stride=1, padding=1)  # 使用padding=1进行卷积操作,padding=1表示在输入张量的边缘添加一圈0，以保持输出张量的空间尺寸不变
print(output3)  # 输出卷积结果