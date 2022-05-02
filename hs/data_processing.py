import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import spectral as spy
from pathlib import Path
from typing import Tuple
import pywt
from scipy.spatial.distance import cdist
import os


def open_image(image_folder: Path) -> Tuple[spy.io.envi.SpectralLibrary, list]:
    cap = Path(image_folder).joinpath('capture')
    if len(list(cap.glob('*raw'))) > 0:
        raw_file = list(cap.glob('*raw'))[0]
        hdr_file = list(cap.glob('*hdr'))[0]
    spec_img = spy.io.envi.open(hdr_file.as_posix(), raw_file.as_posix())
    return spec_img[:, :, :]





def save_image(dir, name, img):
    plt.imsave(dir + f'{name}.png', img)


def ord_frame_list(folder: str):
    frame_list = sorted(
        [(int((filepath.split('frame')[1]).split('.hs')[0]), filepath) for filepath in glob.iglob(folder + '/*.npz')])
    return frame_list


def reconstruct_image(frames: list):
    img = []
    for ind, filepath in frames:
        file = np.load(filepath)
        img.append((file['x'][:, :, :]))  # [channels, height, width]
    img = np.dstack(img)
    return img


def concat_images(folder_list):
    im_list = []
    for folder, frame in folder_list:
        print(folder)
        print(frame)
        img = reconstruct_image(ord_frame_list(folder))
        img = img[:, :, frame[0]:frame[1]]
        print(frame[0], frame[1])
        im_list.append(img)
    image = np.concatenate(im_list, axis=2)
    return image



'''Drawing polygons from csv
inputs are pandas df of polygons and one channel image
output is he mask of contours with the shape of the image'''

def json_to_df(json_path):
    data = json.load(open(json_path))
    df = pd.DataFrame(data['shapes'])
    df = df.loc[:, ['label', 'points']]
    return df


def json_to_csv(json_path, file_name):
    data = json.load(open(json_path))
    df = pd.DataFrame(data['shapes'])
    df = df.loc[:, ['label', 'points']]
    df.to_csv(file_name, index=None)  # Path where the new CSV file will be stored\New File Name.csv
    # return df


def parse_polygons(row):
    points_str = row[2:-2].split('], [')
    points_arr = np.array([np.fromstring(s[:-1], dtype=float, sep=',') for s in points_str])
    # print('points_arr', ((points_arr.reshape(-1, 1, 2))))
    return points_arr


def poly_to_df(csv_path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ['label', 'points']]
    for i, row in df.iterrows():
        row[1] = parse_polygons(row[1])
    return df


'''Create binary masks. Takes an image and dataframe after the function poly_to_df'''


def draw_conts(df, img):
    contours = []
    img = np.zeros((img.shape[0], img.shape[1]))
    for i, row in df.iterrows():
        contour = np.array(row[0]).astype(np.int)[np.newaxis]
        contours.append(contour)
    for c in contours:
        accuracy = 0.0001 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, accuracy, True)
        conts = cv.drawContours(img, [approx], -1, (1, 0, 0), -1)
        print(conts)
    return conts









def create_mask_from_data(img_fold: str, json_path: str, dir, item: str):
    img = open_image(img_fold)
    print(img.shape)
    df = json_to_df(json_path)
    df = df.set_index('label')
    df = df.loc[[item], ['points']]
    mask = draw_conts(df, img)
    # np.save(dir + f'parsimon_class_3_{item}_mask', mask)
    plt.imshow(mask), plt.show()










def calc_area_dist(img_area, img_area2):
    img_area = img_area.reshape(img_area.shape[0] * img_area.shape[1], -1).T
    img_area2 = img_area2.reshape(img_area2.shape[0] * img_area2.shape[1], -1).T
    for i in range(840):
        img_area = np.median(img_area[:, i], axis=0)
        img_area2 = np.median(img_area2[:, i], axis=0)
    dist = np.linalg.norm(img_area - img_area2)
    return dist

img_fold="E:\Parsimon_19_11_20_before\Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19"
image = open_image(img_fold)
image2 = image
# np.save(dir + f'parsimon_class_3_img', img)

fungus_mask = np.load("E:\Vnir_parsimon_class_3_fungus_mask.npy")
fungus_mask2 = np.load("E:\Vnir_parsimon_class_3_fungus_mask.npy")
fruit_mask = np.load("E:\Vnir_parsimon_class_3_fruit_mask.npy")
crown_mask = np.load("E:\Vnir_parsimon_class_3_crown_mask.npy")
fungus2_mask = np.load("E:\Vnir_parsimon_class_3_fungus2_mask.npy")
stage2_mask = np.load("E:\Vnir_parsimon_class_3_stage2_mask.npy")

"""reshape image"""
# img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])


"""subtract fungus and crown"""
fruit_mask = fruit_mask - crown_mask #- fungus_mask #- fungus2_mask# - stage2_mask
"""reshape fruit mask and fungus mask"""
# fruit_mask = fruit_mask.reshape(fruit_mask.shape[0] * fruit_mask.shape[1])
# fungus_mask = fungus_mask.reshape(fungus_mask.shape[0] * fungus_mask.shape[1])
# print(img.shape)
"""img concat"""
# img = np.stack([np.multiply(img[:, i], fruit_mask[0]) for i in range(200)])


def segment_img(img, mask):
    for i in range(img.shape[2]):
        shape = img.shape
        img[:, :, i] = img[mask == 1, i]
    return img.reshape(shape), mask
img_w_masks = np.concatenate((img, fungus_mask2, fruit_mask, crown_mask))
# inv_fungus =
fruit_img, fruit_mask = segment_img(image, fruit_mask)
fruit_img, inv_fungus_mask = segment_img(fruit_img, (abs(1 - fungus_mask)))
fung_img, fungus_mask = segment_img(image, fungus_mask)
plt.figure()
plt.imshow(fruit_img[:, :, 400]), plt.show()
plt.figure()

plt.imshow(fung_img[:, :, 400]), plt.show()



def find_reflectance(hs_img):
    return (np.max(hs_img, axis=2) - hs_img) / np.max(hs_img, axis=2) - np.min(hs_img, axis=2)


"""remove all background pixels(equal to 0 or 1)"""
'''
print(img.shape)
print(np.min(img))
fruit_mask = fruit_mask[fruit_mask != 0]
fruit_mask = fruit_mask[fruit_mask != -1]
fungus_mask = fungus_mask[fungus_mask != 0]
fungus_mask = fungus_mask[fungus_mask != -1]
print(fruit_mask.shape)
print(np.min(fruit_mask))

'''