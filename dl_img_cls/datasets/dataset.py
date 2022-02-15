import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class IMG_DATASET(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []
        if self.train:
            file_path = os.path.join(self.root, 'train.txt')
        else:
            file_path = os.path.join(self.root, 'test.txt')
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            img_path, target = os.path.join(self.root, line.split('\t')[0]), line.split('\t')[1]
            img = Image.open(img_path).convert('RGB')
            self.data.append(img)
            self.targets.append(target)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
