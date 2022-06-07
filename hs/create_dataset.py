import glob
import os
import pathlib as Path
from typing import Tuple
from typing import Union, Iterable
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from skimage import segmentation
import argparse
import json
import hs.hs_data_utils as hsu

parser = argparse.ArgumentParser(description='Arguments for home assignment selection')
parser.add_argument('--path_to_csv', help='Path to a csv file')
parser.add_argument('--path_to_image', help="path to csv file")
parser.add_argument('--directory', default='/Users/ramtahor/Desktop/parsimmon_dataset/',
                    help='Directory where the file would be saved')

path = '/Users/ramtahor/Desktop/hand/_Vnir-Parsimons_class-2_51-60_no-leaves_2020-11-19_12-39-37'
poly_path = '/Users/ramtahor/Desktop/Parsimon-PNG/_Vnir-Parsimons_class-2_51-60_no-leaves_2020-11-19_12-39-37.json'
# TODO:
# open hs image
# crop each fruit
# save by the label and image name







def json_to_df(json_path):
    data = json.load(open(json_path))
    df = pd.DataFrame(data['shapes'])
    df = df.loc[:, ['label', 'points']]
    return df


def get_name(name_list, label: str):
    number = str(int(name_list[3].split('-')[0]) - 1 + int(label.split('_')[1]))
    f_class = name_list[2]
    time = name_list[5]
    return f_class + '_' + number + '_' + time


def arrange_df(json_path: str, df: pd.DataFrame):
    name_list = json_path.split('_')
    df['name'] = None
    for i, row in df.iterrows():
        row['name'] = get_name(name_list, row['label'])
    return df
df = arrange_df(poly_path,json_to_df(poly_path))
print('df is:', df.columns)
print(df)

def cut_conts(df, hs_img: np.array()):
    mask_img = np.zeros(hs_img.shape[0, 1])
    for i, row in df.iterrows():
        contour = np.array(row[1]).astype(np.int)[np.newaxis]

        accuracy = 0.0001 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, accuracy, True)
        cont = cv.drawContours(mask_img, [approx], -1, (1, 0, 0), -1)
    return cont
