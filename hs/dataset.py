import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
from pathlib import Path
from typing import Union, Iterable
from collections import defaultdict
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

path = '/Users/ramtahor/Desktop/new_dataset'


class HsDataset(Dataset):
    def __init__(self, images_path, method):
        self.images, self.labels = self.get_image_and_label(images_path, [400], method)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])

    def get_image_and_label(self, directory: str, band: Union[int, Iterable], method: str):
        im_list = []
        label_list = []
        label = {'class': 0, 'number': 1, 'date': 2}
        i = label[method]
        for file in Path(directory).glob('*/*.npy'):
            image = np.load(str(file))[:, :, band]
            img = iaa.Sequential([
                iaa.Resize({"height": 224, "width": 224}),  # resize image
                # iaa.AdditiveGaussianNoise(),  # crop images from each side by 0 to 16px (randomly chosen)
                iaa.ScaleX((0.5, 1.5)),  # rescale along X axis
                iaa.ScaleY((0.5, 1.5)),  # rescale along Y axis
                iaa.Rotate((-45, 45)),  # rotate randomly in -45 45 degrees
                iaa.Fliplr(0.5),  # horizontally flip 50% of the images
                iaa.GaussianBlur(sigma=(0, 3.0))])  # blur images with a sigma of 0 to 3.0

            image = img(images=(image.reshape(image.shape[2], image.shape[0], image.shape[1])))
            im_list.append(image.astype(np.float16))
            label_list.append(int(((str(file).split('.npy')[0]).split('class-')[1]).split('_')[i]))
        return np.array(im_list), np.array(label_list)

    def get_dataloaders(self, batch_size, train_split=0.8):
        idx = np.arange(len(self.images))
        train_size = int(len(self.images) * train_split)
        train_idx = idx[:train_size]
        val_idx = idx[train_size:]
        train_dl = DataLoader(dataset=Subset(self, indices=train_idx), shuffle=True, batch_size=batch_size)
        val_dl = DataLoader(dataset=Subset(self, indices=val_idx), batch_size=batch_size)
        return train_dl, val_dl


data = HsDataset(path, 'class')
train_loader, test_loader = data.get_dataloaders(batch_size=5)
for i in range(len(data)):
    plt.imshow((data[i][0].reshape(224, 224)).astype(np.uint16)), plt.show()
