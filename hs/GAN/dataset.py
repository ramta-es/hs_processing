import rasterio.plot
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
from pathlib import Path

import imgaug
from collections import defaultdict
import imgaug.augmenters as iaa
import random
import torch
import hs.data_processing as hsu

class HsDataset(Dataset):
    def __init__(self, root, json_path_before, json_path_after):
        self.root = Path(root)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.polygons_pred = defaultdict(list)  # image name: list of shapely polygons
        self.polygons_true = defaultdict(list)  # image name: list of shapely polygons
        # Should be 10 polygons in total- fruits without crown
        self.parse_polygons(json_path_before, self.polygons_pred, ) # Add label
        self.parse_polygons(json_path_after, self.polygons_true, ) # Add label
        self.images_pred = list(self.root.joinpath('image_path').glob('*.raw')) #replace to correct path to images
        self.images_true = list(self.root.joinpath('image_path').glob('*.raw'))  # replace to correct path to images
        self.seq = iaa.Sequential([
            iaa.CropToFixedSize(width=128, height=128),
        ])

        random.shuffle(self.images)

    def __len__(self):
        return len(self.images_pred)

    def parse_polygons(self, json_path, polygons, label: str):
        df = hsu.json_to_df(json_path)
        for i, row in df.iterrows():
            print(row.points)
            points_str = row.points[10:-2].split(',')
            points_arr = np.array([np.fromstring(s[:-2], dtype=float, sep=' ') for s in points_str])
            try:
                polygons[label].append(imgaug.Polygon((points_arr, i + 1)))
            except Exception:
                print(i, points_str)

    def __getitem__(self, idx):
        # load images and polygon
        img_path_pred = self.images_pred[idx]
        img_path_true = self.images_true[idx]
        img_pred = hsu.open_image(img_path_pred)
        img_pred = hsu.find_reflectance(img_pred)
        img_true = hsu.open_image(img_path_true)
        img_true = hsu.find_reflectance(img_true)
        polygons_pred = imgaug.PolygonsOnImage(polygons=self.polygons[img_path_pred.stem], shape=img_pred.shape)  # array of points (array(4,2))
        polygons_true = imgaug.PolygonsOnImage(polygons=self.polygons[img_path_true.stem], shape=img_true.shape)  # array of points (array(4,2))
        crop_pred, poly_pred = self.seq(image=img_pred, polygons=polygons_pred)  # random factor
        crop_true, poly_true = self.seq(image=img_true, polygons=polygons_true)
        try:
            lbl_pred = int(poly_pred.remove_out_of_image_fraction_(0.95).empty + fruit_idx)
        except: lbl_pred = 0
        pass
        #print(poly1.remove_out_of_image_fraction_(0.1).empty)
        try:
            lbl_true = int(poly_true.remove_out_of_image_fraction_(0.1).empty + fruit_idx)
        except: lbl_true = 0
        pass
        t1 = torch.from_numpy(np.transpose(crop_pred, (2, 0, 1)))
        t2 = torch.from_numpy(np.transpose(crop_true, (2, 0, 1)))
        return t1, t2

    def get_dataloaders(self, batch_size, train_split=0.8):
        idx = np.arange(len(self.images))
        train_size = int(len(self.images)*train_split)
        train_idx = idx[:train_size]
        val_idx = idx[train_size:]
        train_dl = DataLoader(dataset=Subset(self, indices=train_idx), shuffle=True, batch_size=batch_size)
        val_dl = DataLoader(dataset=Subset(self, indices=val_idx), batch_size=batch_size)
        return train_dl, val_dl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json_file_before', type=str, help='path to polygons csv file to parse')
    parser.add_argument('json_file_after', type=str, help='path to polygons csv file to parse')
    parser.add_argument('root', type=str, help='path to images root')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help='add debug prints')
    args = parser.parse_args()
    print(vars(args))

    dataset = HsDataset(root=args.root, csv_path=args.csv_file)
    x = dataset[2]
    #csv_file = pd.read_csv()