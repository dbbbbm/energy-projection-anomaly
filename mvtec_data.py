import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils import data

class MVTec(data.Dataset):
    def __init__(self, transform, root, split, cl):
        super(MVTec, self).__init__()

        self.transform = transform if transform is not None else lambda x: x
        self.file_path = []
        self.label = []

        for dt in os.listdir(os.path.join(root, cl, split)):
            for f in os.listdir(os.path.join(root, cl, split, dt)):
                if dt == 'good':
                    self.label.append(0)
                else:
                    self.label.append(1)
                self.file_path.append(os.path.join(root, cl, split, dt, f))
    
    def __getitem__(self, index):
        img = Image.open(self.file_path[index])
        label = self.label[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.label)


def get_dataloader(root, cl, dtype='train', bs=64, nw=2, img_size=64):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dset = MVTec(transform, root, dtype, cl)
    train_flag = True if dtype == 'train' else False
    dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
                                 drop_last=train_flag, num_workers=nw, pin_memory=True)
    return dataloader


if __name__ == '__main__':
    root = '/media/user/disk/mvtec_ad/'
    train_loader = get_dataloader(root, 0)

    for(x, label) in train_loader:
        print(x)
