import glob
import json
import os
import pathlib as Path
from typing import Tuple
from typing import Union, Iterable

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spectral as spy
import tensorflow as tf
from skimage import segmentation


def open_image(image_folder: Path) -> Tuple[spy.io.envi.SpectralLibrary, list]:
    cap = Path.Path(image_folder).joinpath('capture')
    if len(list(cap.glob('*.raw'))) > 0:
        raw_file = list(cap.glob('*.raw'))[0]
        hdr_file = list(cap.glob('*.hdr'))[0]
    spec_img = spy.io.envi.open(hdr_file.as_posix(), raw_file.as_posix())
    spec_img = spec_img[1]
    spec_img = spec_img[:, :, :]

    return spec_img

def ord_frame_list_all(folder: str):
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


name = 'stram_03_pasul'

"""cuts the image to a 1000x256 pixels smaller images. return a list of the smaller images and their index for later assembly"""

def hs_to_fake_rgb(im: np.ndarray, reduction: Union[str, int, Iterable] = 'norm') -> np.ndarray:
    """
    Converts a HxWxC hyperspectral image into a HxWx3 RGB image.

    :param im: Input hyperspectral image.
    :param reduction: 'norm', channel index, or 3 channel indices.
    :return:
    """
    if reduction == 'norm':
        im = tf.norm(tf.cast(im, tf.float32), axis=0) / np.sqrt(im.shape[0])
    elif isinstance(reduction, int):
        im = im[reduction, :, :]
    elif isinstance(reduction, Iterable):
        im = im[list(reduction), :, :].transpose([1, 2, 0])
    else:
        assert False, "Unsupported reduction"
    return im.numpy()


# Convert hs image into fake RGB image after given a list of channels
def hyper_to_fake_rgb(img, channels: list):
    im = np.zeros((img.shape[1], img.shape[2], len(channels)))
    for i in range(len(channels)):
        im[:, :, i] = np.median(img[channels[i] - 20:channels[i] + 30, :, :], axis=0) / np.max(
            img[channels[i] - 20:channels[i] + 30, :, :])
    return im.astype(np.float64)





def cut_image(img):
    img_list = []
    print(img.shape[2])
    for cut in range(0, img.shape[2], 1000):
        img_list.append((img[:, :, cut:cut + 1000], len(img_list)))
        print(len(img_list))
        print('done')
    return img_list


def make_dir(dir_name):
    # Create directory
    try:
        # Create target Directory
        os.mkdir(dir_name)
        print("Directory ", dir_name, " Created ")
    except FileExistsError:
        print("Directory ", dir_name, " already exists")
    print('dir was created')




def save_cut_image(dir_name, img_list):
    make_dir(dir_name + '_for_tag')
    make_dir(dir_name)
    for img, idx in img_list:
        plt.imsave(dir_name + '_for_tag' + f'\{idx}.png', img[150, :, :])
        np.save(dir_name + f'\{idx}_full', img[:, :, :])
    print('saved')

def save_image(img, name):
    plt.imsave(r"C:\Users\g\Desktop\clementine_seg" + f'\{name}.png', img[150, :, :])




def json_to_csv(json_path, file_name):
    data = json.load(open(json_path))
    df = pd.DataFrame(data['shapes'])
    df = df.loc[:, ['label', 'points']]
    df.to_csv(file_name, index=None)  # Path where the new CSV file will be stored\New File Name.csv
    # return df


def json_to_df(json_path):
    data = json.load(open(json_path))
    df = pd.DataFrame(data['shapes'])
    df = df.loc[:, ['label', 'points']]
    return df


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


'''Create binary masks. Takes an image and polygons dataframe 

and draws the mask

'''


def draw_conts(json_path, hs_img_shape):
    df = json_to_df(json_path)
    contours = []
    img = np.zeros(hs_img_shape)
    for i, row in df.iterrows():
        contour = np.array(row[1]).astype(np.int)[np.newaxis]
        contours.append(contour)
    for c in (contours):
        accuracy = 0.0001 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, accuracy, True)
        conts = cv.drawContours(img, [approx], -1, (1, 0, 0), -1)
    return conts


"""
Reshapes the image to Pixel x Channels
"""
def img_reshape(img, mask):
    shape = img.shape
    img = img.reshape(-1, img.shape[1] * img.shape[2]).T
    mask = mask.reshape(mask.shape[0] * mask.shape[1]).T

    img_df = pd.DataFrame(img)
    img_df['label'] = mask
    return img_df, shape


def fit_new_data(folder):
    frames = ord_frame_list_all(folder)
    img = reconstruct_image(frames)
    img = img[:, :, :]
    img_shape = img.shape
    im_df = img.reshape(-1, img.shape[1] * img.shape[2]).T
    im_df = pd.DataFrame(im_df)
    return im_df, img, img_shape


def plot_clf_results(orig_im, img_mask, clf_type):
    plt.figure()
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 8))
    ax[0].imshow(segmentation.mark_boundaries(orig_im[100, :, :], img_mask, color=2, mode='thick'))
    ax[0].set_title(f'Image, mask and segmentation boundaries {clf_type}')
    ax[1].imshow(img_mask)
    ax[1].set_title('Segmentation')
    fig.tight_layout()
    ax[2].imshow(orig_im[100, :, :])
    ax[2].set_title('Original image')




def create_small_images(**kwargs):
    img = reconstruct_image(frames=ord_frame_list_all(kwargs['path_to_frame_forlde']))
    save_cut_image(kwargs['path_and_name_of_dir'], cut_image(img))


""" Get median value for every row of a 2d image (one channel) 
"""
def get_median(img_2d, mask, shape):  # 2d imges 256x width
    med = np.zeros((shape[2], 1))
    for i in range(img_2d.shape[0]):
        med[i] = np.median(img_2d[i][mask[i] != 0])
    return med  # 256x1

"""Calc the median for the whole image
"""

def get_median_dark(img_2d):  # 2d imges 256x width
    med = np.zeros((256, 1))
    for i in range(img_2d.shape[0]):
        med[i] = np.median(img_2d[i])
    return med  # 256x1

# find the median value for every channel in the hs image and arrange it in an array
def find_white_ref(img, mask):
    im = np.zeros((img.shape[0], img.shape[1], 1))
    for i in range(img.shape[0]):
        im[i] = get_median(img[i, :, :], mask)
    return im.reshape((224, 256))


def find_dark_ref(img):
    im = np.zeros((img.shape[0], img.shape[1], 1))
    for i in range(img.shape[0]):
        im[i] = get_median_dark(img[i, :, :])
    return im.reshape((224, 256))





