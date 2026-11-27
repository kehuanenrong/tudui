import os
import torchvision

train_data = torchvision.datasets.CIFAR10(root="cifar10", train=True, download=True)
test_data = torchvision.datasets.CIFAR10(root="cifar10", train=False, download=True)

# 保存数据的文件夹名称
folder_path = "custom_data"

# 训练数据保存到 train 文件夹
train_folder_path = os.path.join(folder_path, "train") # 路径拼接，but不负责创建文件夹
# 测试数据保存到 test 文件夹
test_folder_path = os.path.join(folder_path, "test")

if not os.path.exists(train_folder_path):
    os.makedirs(train_folder_path)
if not os.path.exists(test_folder_path):
    os.makedirs(test_folder_path)

# 0 代表 airplanes，1 代表 cars，以此类推
# airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks
cifar_num2item = {0: "airplanes", 1: "cars",
                    2: "birds", 3: "cats",
                    4: "deer", 5: "dogs",
                    6: "frogs", 7: "horses",
                    8: "ship", 9: "trucks"}

# 保存 50 张训练图片到 train 文件夹
for i in range(50):
    img = train_data[i][0]  # 获取图片数据
    target = train_data[i][1]  # 获取图片标签
    target = cifar_num2item[target]  # 将数字标签转换为类别名称
    img.save(os.path.join(train_folder_path, f"{i}_{target}.png"))

# 保存 10 张测试图片到 test 文件夹
for i in range(10):
    img = test_data[i][0]  # 获取图片数据
    target = test_data[i][1]  # 获取图片标签
    target = cifar_num2item[target]  # 将数字标签转换为类别名称
    img.save(os.path.join(test_folder_path, f"{i}_{target}.png"))