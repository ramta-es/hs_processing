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

root = '/Volumes/My Passport'


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


def create_fourier_for_pixel(img_fold: str, pixel: tuple, color: str):
    img = open_image(img_fold)
    pix = img[pixel[0], pixel[1], 300:600] #pixel = (y_coord, x_coord)

    print(pix.shape)
    # pix = np.concatenate((pix, pix, pix, pix, pix, pix), axis=0)
    # pix = np.stack([pix, pix], axis=1)
    # print(pix.shape[0])
    ft = np.fft.rfft(pix)
    print(ft)

    # t = np.arange(pix)
    # t = pix
    # sp = np.fft.fft(np.sin(t))
    # freq = np.fft.fftfreq(t.shape[-1])
    # plt.plot(freq, sp.real, freq, sp.imag)


    print(ft.shape)
    # np.save(dir + f'parsimon_class_3_{item}_mask', mask)
    # print('ft', (list(ft.real)))
    # plt.plot(list(range(396)), list(ft.real), color=color)
    # plt.imshow(ft), \
    plt.show()
    return abs(ft)








def calc_dist(*args, idx):
    dist = [f'idx = {idx}']
    for i in range(len(args)):
        if i != idx:
            dist.append(np.linalg.norm(args[idx] - args[i]))
    return dist

def calc_area_dist(img_area, img_area2):
    img_area = img_area.reshape(img_area.shape[0] * img_area.shape[1], -1).T
    img_area2 = img_area2.reshape(img_area2.shape[0] * img_area2.shape[1], -1).T
    for i in range(840):
        img_area = np.median(img_area[:, i], axis=0)
        img_area2 = np.median(img_area2[:, i], axis=0)
    dist = np.linalg.norm(img_area - img_area2)
    return dist

# im = np.load("E:\Vnir_parsimon_class_3_fungus_mask.npy")
# im2 = np.load("E:\Vnir_parsimon_class_3_fruit_mask.npy")
# plt.imshow(im), plt.show()
# plt.imshow(im2), plt.show()

# create_mask_from_data(img_fold="E:\Parsimon_19_11_20_before\Swir_Parsimon__class_3_71_80_no_leaves__2020-11-22_10-00-49", json_path="E:\Swir_Parsimon__class_3_71_80_no_leaves_3_2020-11-22_10-00-49_labeled.json", dir=r'E:', item='furit2')
#
# fungus = create_fourier_for_pixel(img_fold="E:\Parsimon_19_11_20_before\Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19", pixel=(1595, 1062), color='black')
# fruit1 = create_fourier_for_pixel(img_fold="E:\Parsimon_19_11_20_before\Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19", pixel=(3229, 1030), color='red')
# fruit2 = create_fourier_for_pixel(img_fold="E:\Parsimon_19_11_20_before\Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19", pixel=(1945, 554), color='blue')
# crown = create_fourier_for_pixel(img_fold="E:\Parsimon_19_11_20_before\Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19", pixel=(422, 410), color='green')
# crown2 = create_fourier_for_pixel(img_fold="E:\Parsimon_19_11_20_before\Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19", pixel=(1074, 386), color='green')
# fruit3 = create_fourier_for_pixel(img_fold="E:\Parsimon_19_11_20_before\Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19", pixel=(1577, 179), color='green')
# fungus2 = create_fourier_for_pixel(img_fold="E:\Parsimon_19_11_20_before\Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19", pixel=(281, 367), color='black')
# fungus3 = create_fourier_for_pixel(img_fold="E:\Parsimon_19_11_20_before\Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19", pixel=(3103, 235), color='black')
# fungus4 = create_fourier_for_pixel(img_fold="E:\Parsimon_19_11_20_before\Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19", pixel=(2219, 1164), color='black')

d = {'fruit_1': (1070, 1259), 'fruit_2': (1128, 1945), 'fruit_3': (1276, 2515), 'fruit_4': (117, 1598), 'fruit_5': (615, 2277), 'crown_1': (1208, 2306), 'crown_2': (1141, 1653), 'fungus_1': (1058, 1740), 'fungus_2': (1169, 2219), 'fungus_3': (1099, 3042), 'fungus_4': (257, 2470), 'fungus_5': (341, 2530)}

img_fold = '/Volumes/My Passport/Parsimon_19_11_20_before/Vnir_Parsimons__class_3_71_80_no_leaves__2020-11-19_13-00-19'
img = open_image(img_fold)

def calc_wt(pix):
    db = pywt.Wavelet('db1')
    return pywt.wavedec(pix, db)

def calc_pixel_dist(img, pix: tuple, pix2: tuple):
    pix1 = img[pix[1], pix[0], 0:200] #pixel = (y_coord, x_coord)
    pix1 = calc_wt(pix1)
    pix2 = img[pix2[1], pix2[0], 0:200] #pixel = (y_coord, x_coord)
    pix2 = calc_wt(pix2)
    # print(pix1.shape)
    # print(pix2.shape)
    return np.linalg.norm(pix2[0] - pix1[0])


def calc_mahalanobis_dist(img, pix1, pix2):
    pix1 = img[pix1[1]:pix1[1] + 1, pix1[0]:pix1[0] + 1, 600:]  # pixel = (y_coord, x_coord)
    pix1 = pix1.reshape(pix1.shape[0] * pix1.shape[1], pix1.shape[2]).T
    pix2 = img[pix2[1]:pix2[1] + 1, pix2[0]:pix2[0] + 1, 600:]  # pixel = (y_coord, x_coord)
    pix2 = pix2.reshape(pix2.shape[0] * pix2.shape[1], pix2.shape[2]).T

    results = cdist(pix1, pix2, 'mahalanobis')
    print(results.shape)

    return np.linalg.norm(np.diag(results))


# dist1 = calc_mahalanobis_dist(img, d['fruit_1'], d['fruit_2'])

# print(np.linalg.norm(dist1))



dist1 = (calc_mahalanobis_dist(img, d['fruit_1'], d['fruit_2']), 'green')
dist2 = (calc_mahalanobis_dist(img, d['fruit_1'], d['fruit_3']), 'green')
dist3 = (calc_mahalanobis_dist(img, d['fruit_2'], d['fruit_3']), 'green')
dist4 = (calc_mahalanobis_dist(img, d['fruit_4'], d['fruit_3']), 'green')
dist5 = (calc_mahalanobis_dist(img, d['fruit_4'], d['fruit_2']), 'green')
dist6 = (calc_mahalanobis_dist(img, d['fruit_4'], d['fruit_5']), 'green')
dist7 = (calc_mahalanobis_dist(img, d['fruit_5'], d['fruit_1']), 'green')
dist8 = (calc_mahalanobis_dist(img, d['fruit_4'], d['fruit_3']), 'green')
dist9 = (calc_mahalanobis_dist(img, d['fruit_2'], d['fruit_5']), 'green')
dist10 = (calc_mahalanobis_dist(img, d['fruit_5'], d['fruit_3']), 'green')

#
dist11 = (calc_mahalanobis_dist(img, d['fungus_4'], d['fungus_3']), 'red')
dist12 = (calc_mahalanobis_dist(img, d['fungus_5'], d['fungus_2']), 'red')
dist13 = (calc_mahalanobis_dist(img, d['fungus_1'], d['fungus_2']), 'red')
dist14 = (calc_mahalanobis_dist(img, d['fungus_1'], d['fungus_3']), 'red')
dist15 = (calc_mahalanobis_dist(img, d['fungus_2'], d['fungus_3']), 'red')
dist16 = (calc_mahalanobis_dist(img, d['fungus_2'], d['fungus_2']), 'red')
dist17 = (calc_mahalanobis_dist(img, d['fungus_3'], d['fungus_5']), 'red')
dist18 = (calc_mahalanobis_dist(img, d['fungus_4'], d['fungus_1']), 'red')
dist19 = (calc_mahalanobis_dist(img, d['fungus_5'], d['fungus_2']), 'red')
dist20 = (calc_mahalanobis_dist(img, d['fungus_4'], d['fungus_2']), 'red')


dist21 = (calc_mahalanobis_dist(img, d['fruit_1'], d['fungus_1']), 'black')
dist22 = (calc_mahalanobis_dist(img, d['fruit_1'], d['fungus_2']), 'black')
dist23 = (calc_mahalanobis_dist(img, d['fruit_1'], d['fungus_3']), 'black')
dist24 = (calc_mahalanobis_dist(img, d['fruit_2'], d['fungus_1']), 'black')
dist25 = (calc_mahalanobis_dist(img, d['fruit_2'], d['fungus_2']), 'black')
dist26 = (calc_mahalanobis_dist(img, d['fruit_2'], d['fungus_3']), 'black')
dist27 = (calc_mahalanobis_dist(img, d['fruit_3'], d['fungus_1']), 'black')
dist28 = (calc_mahalanobis_dist(img, d['fruit_3'], d['fungus_2']), 'black')
dist29 = (calc_mahalanobis_dist(img, d['fruit_3'], d['fungus_3']), 'black')
dist30 = (calc_mahalanobis_dist(img, d['fungus_3'], d['fruit_5']), 'black')


#

l1 = [dist1[0], dist2[0], dist3[0], dist4[0], dist5[0], dist5[0], dist7[0], dist8[0], dist9[0], dist10[0]]
l2 = [dist11[0], dist12[0], dist13[0], dist14[0], dist15[0], dist16[0], dist17[0], dist18[0], dist19[0], dist20[0]]
l3 = [dist21[0], dist22[0], dist23[0], dist24[0], dist25[0], dist26[0], dist27[0], dist28[0], dist29[0], dist30[0]]

x = [i for i in range(10)]

plt.scatter(x, l1, color='green')
plt.scatter(x, l2, color='red')
plt.scatter(x, l3, color='black')
# print(np.linalg.norm(l[i][0]))
# print(l)
plt.title('Euclidian norm for Mahalanobis distance on the wavelet tranform on fruit and fungus pixels')
plt.grid(axis='y')
plt.xlabel('point index')
plt.ylabel('mahalanobis diag norm')
plt.legend(['fr2fr', 'fu2fu', 'fr2fu'])
plt.show()






# dist_fruit_fruit = np.linalg.norm(fruit1 - fruit2)
# dist_fruit_fruit3 = np.linalg.norm(fruit3 - fruit2)
# dist_fruit_fruit2 = np.linalg.norm(fruit1 - fruit3)
# dist_fruit_fungus = np.linalg.norm(fruit1 - fungus)
# dist_fruit2_fungus = np.linalg.norm(fruit2 - fungus)
# dist_fruit_crown = np.linalg.norm(fruit1 - crown)
# dist_fruit_crown2 = np.linalg.norm(fruit1 - crown2)
# dist_crown_fungus = np.linalg.norm(crown - fungus)
# dist_crown_fungus2 = np.linalg.norm(crown2 - fungus)

# norm_fruit = np.linalg.norm(fruit1)
# norm_fruit3 = np.linalg.norm(fruit3)
# norm_fruit2 = np.linalg.norm(fruit2)
# norm_fungus = np.linalg.norm(fungus)
# norm_crown = np.linalg.norm(crown)
# norm_crown2 = np.linalg.norm(crown2)
#
#
# print('dist_fruit_fruit', dist_fruit_fruit)
# print('dist_fruit2_fungus', dist_fruit2_fungus)
# print('dist_fruit_fungus', dist_fruit_fungus)
# print('dist_fruit_crown', dist_fruit_crown)
# print('dist_crown_fungus', dist_crown_fungus)
# print('dist_fruit_fruit2', dist_fruit_fruit2)
# print('dist_crown_fungus2', dist_crown_fungus2)
# print('dist_fruit_fruit3', dist_fruit_fruit3)

# print('norm_fruit', norm_fruit)
# print('norm_fruit2', norm_fruit2)
# print('norm_fruit3', norm_fruit3)
# print('norm_fungus', norm_fungus)
# print('norm_crown', norm_crown)
# print('norm_crown2', norm_crown2)
#




# df = json_to_df("E:\Swir_Parsimon__class_3_71_80_no_leaves_3_2020-11-22_10-00-49_labeled.json")
# df2 = df.loc[:, ['label']]
# df2 = df.set_index('label')
# df2 = df2.loc[['fungus'], ['points']]
# # df = df
# #
#
# #
# #
# print('head', df2.head)



