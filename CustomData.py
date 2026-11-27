import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root=None, train=True):
        if train:
            self.folder_path = os.path.join(root, "train")
        else:
            self.folder_path = os.path.join(root, "test")
        self.files_list = os.listdir(self.folder_path)

    def __getitem__(self,index):
        img_name = self.files_list[index]
        img_path = os.path.join(self.folder_path, img_name)
        img = Image.open(img_path)
        target = img_name.split('.png')[0].split('_')[-1]
        return [img, target]

    def __len__(self):
        return len(self.files_list)

# dataset = CustomDataset()
# print(dataset[0])

train_dataset = CustomDataset(root="custom_data", train=True)
test_dataset = CustomDataset(root="custom_data", train=False)
print(train_dataset[0])
print(test_dataset[0])