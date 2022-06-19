import rasterio.plot
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
from pathlib import Path
from typing import Union, Iterable

import os
import imgaug
from collections import defaultdict
import imgaug.augmenters as iaa
import random
import torch
import matplotlib.pyplot as plt
path = '/Users/ramtahor/Desktop/new_dataset'

class HsDataset(Dataset):
    def __init__(self, images_path, method):
        self.images, self.labels = self.get_image_and_label(images_path, [400], method)

        seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Rotate(())
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0

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
            img = iaa.Resize({"height": 500, "width": 500})
            image = img(images=(image.reshape(image.shape[2], image.shape[0], image.shape[1])))
            im_list.append(image)
            label_list.append(((str(file).split('.npy')[0]).split('class-')[1]).split('_')[i])
        return np.array(im_list), np.array(label_list)


    def get_dataloaders(self, batch_size, train_split=0.8):
        idx = np.arange(len(self.images))
        train_size = int(len(self.images)*train_split)
        train_idx = idx[:train_size]
        val_idx = idx[train_size:]
        train_dl = DataLoader(dataset=Subset(self, indices=train_idx), shuffle=True, batch_size=batch_size)
        val_dl = DataLoader(dataset=Subset(self, indices=val_idx), batch_size=batch_size)
        return train_dl, val_dl


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('json_file_before', type=str, help='path to polygons csv file to parse')
#     parser.add_argument('json_file_after', type=str, help='path to polygons csv file to parse')
#     parser.add_argument('root', type=str, help='path to images root')
#     parser.add_argument('-d', '--debug', default=False, action='store_true', help='add debug prints')
#     args = parser.parse_args()
#     print(vars(args))
#
#     dataset = HsDataset(root=args.root, csv_path=args.csv_file)
#     x = dataset[2]
#     #csv_file = pd.read_csv()

data = HsDataset(path, 'class')
print(data[100][0].shape)
plt.imshow(data[100][0].reshape(data[100][0].shape[1], data[100][0].shape[2])), plt.show()







