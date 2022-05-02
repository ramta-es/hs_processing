import pickle
from pathlib import Path
from collections import OrderedDict
import cv2 as cv
import numpy as np
import pathlib as Path
import matplotlib.pyplot as plt
import spectral as spy
from typing import Tuple
import os

labelme_line = 'labelme <image to label> -O apc2016_obj3.json'
'Swir_Parsimon_side_class_2_51_60_no_leaves__after'
'Vnir_Parsimons__class_1_31_40_no_leaves__after'
# dir = '/Volumes/My Passport/Parsimon_17_3_21_after'
dir = '/Volumes/My Passport/Parsimon_19_11_20_before'
sub_dir = 'Vnir_Parsimons__class_0_11_20_no_leaves_2_2020-11-19_12-03-26'

'/Volumes/My Passport/Parsimon_19_11_20_before/Swir_Parsimon__class_3_71_80_no_leaves__2020-11-22_10-00-49/Swir_Parsimon__class_3_71_80_no_leaves_3_2020-11-22_10-00-49.png'


####################################################
####################################################

def open_image(image_folder: Path) -> Tuple[spy.io.envi.SpectralLibrary, list]:
    print('opening image')
    im_time = 'before'
    if '17_3_21' in image_folder:
        im_time = 'after'
    cap = Path.Path(image_folder).joinpath('capture')
    if len(list(cap.glob('*.raw'))) > 0:
        raw_file = list(cap.glob('*.raw'))[0]
        hdr_file = list(cap.glob('*.hdr'))[0]
    spec_img = spy.io.envi.open(hdr_file.as_posix(), raw_file.as_posix())
    spec_img = spec_img[:, :, :]

    return spec_img, im_time


def get_folder_and_class(in_path):
    if 'Vnir_Parsimons' in in_path:
        class_ = int((in_path.split('Vnir_Parsimons__')[1]).split('_')[1])
        image_folder = (in_path.split('/Vnir_Parsimons')[0] + '/Vnir_Parsimons' + in_path.split('/Vnir_Parsimons')[1])
        chan = 840
    else:
        class_ = int(in_path.split('_class_')[1].split('_')[0])
        image_folder = (in_path.split('/Swir_Parsimon')[0] + '/Swir_Parsimon' + in_path.split('/Swir_Parsimon')[1])
        chan = 256

    return image_folder, class_, chan


def open_image_in_dict(key):
    print('key', key)
    image_folder, class_, chan = get_folder_and_class(key)
    img, im_time = open_image(image_folder=image_folder)
    # open the image
    img = img[:, :, :]
    return img, im_time, chan, class_


# creates a crop from the rectangle coordinates and put it the rectangle list
def cut_crop(rect_list, img):
    for i in range(len(rect_list)):
        rect_list[i] = (
            img[rect_list[i][1]:rect_list[i][3] + rect_list[i][1], rect_list[i][0]:rect_list[i][2] + rect_list[i][0], :])
    return rect_list


def make_crops_before_after(rect_dict):
    out_dict = {}
    for key in rect_dict:
        img, im_time, channels, class_ = open_image_in_dict(key)
        crop_list = cut_crop(rect_dict[key], img)
        out_dict[key] = crop_list
    return out_dict, channels, im_time, class_


########################
########################

# calculate the mathematical operator over the crop
# Gets a list of numpy arrays as an input and outputs a list of lists
# channel length after the mathematical operator
def ope(input_list, channels, operator):
    print(f'doing {operator}')
    out_list = []
    for i in range(len(input_list)):
        out_list.append([operator(input_list[i][:, :, j]) for j in range(channels)])
    return out_list  # outputs a list of arrays


def flat(in_list, channels, operator):  # calculate the mathematical operator over the crop
    print('doing flat')
    out_list = []
    for i in range(len(in_list)):
        out_list.append([(in_list[i][:, :, j]) for j in range(channels)])
    return out_list  # outputs a list of arrays


def inlist_avg(in_list, channels):  # list of arrays
    print('doing inlist_avg')
    tmp_list = []
    for i in range(channels):
        tmp_list.append((sum([(item[i]) for item in in_list]) / len(in_list)))
    return tmp_list


def make_plot(y_list, channels, line, color, color2, class_):
    x_list = [i for i in range(channels)]
    if line == 'solid' and color != color2:
        plt.plot(x_list, y_list, linestyle=line, color=color, label=f'class{class_}')
    else:
        plt.plot(x_list, y_list, linestyle=line, color=color)
    plt.grid()
    plt.xlabel('Channel')
    plt.ylabel('Power')
    plt.title('Variance Vnir before after')
    plt.legend()
    # plt.show()


def make_plot_new_format(y_list, channels, line, color, class_, state):
    x_list = [i for i in range(channels)]
    plt.plot(x_list, y_list, linestyle=line, color=color, label=f'{state}')
    plt.grid()
    plt.xlabel('Channel')
    plt.ylabel('Power')
    plt.title(f' Vnir class{class_}')
    plt.legend()
    # plt.show()


def check_line_and_color(key):
    if 'Swir_Parsimons' in key:
        class_ = int((key.split('Swir_Parsimons__')[1]).split('_')[1])
    else:
        class_ = int(key.split('_class_')[1].split('_')[0])
    color_map = {0: 'cornflowerblue', 1: 'lightcoral', 2: 'springgreen', 3: 'darkorange'}
    if '17_3_21' in key:
        state = 'after'
    else:
        state = 'before'
    time_map = {'before': 'solid', 'after': 'dashdot'}

    return time_map[state], color_map[class_]


def check_line_and_color_sick_healthy(key):
    if 'Swir_Parsimons' in key:
        class_ = int((key.split('Swir_Parsimons__')[1]).split('_')[1])
    else:
        class_ = int(key.split('_class_')[1].split('_')[0])
    color_map = {0: 'cornflowerblue', 1: 'lightcoral', 2: 'springgreen', 3: 'darkorange'}
    if '17_3_21' in key:
        state = 'after'
    else:
        state = 'before'
    time_map = {'before': 'solid', 'after': 'dashdot'}

    return time_map[state], color_map[class_]










def open_main_dict(in_dict):
    out_dict = {}
    for key in in_dict:
        out_dict[key], channels, time, class_ = make_crops_before_after(in_dict[key], 256)
        print('Channels', channels)
        # avg = ope_func(crops0, channels, operator=np.average)
    color2 = 'none'
    print(out_dict.keys())
    for key in out_dict:
        for key2 in out_dict[key]:
            line, color = check_line_and_color(key)
            make_plot(out_dict[key2], channels, color=color, line=line, color2=color2)
            color2 = color
    plt.title('Average Swir before after')
    plt.xlabel('Channel')
    plt.ylabel('Power')
    plt.grid()
    plt.legend()
    plt.show()


def data_converter(in_tupel):
    y, z = in_tupel[1], in_tupel[2]
    for i in range(len(in_tupel[1])):
        z[i] = [int(j) for j in y[i]]
    for i in range(len(in_tupel[2])):
        y[i] = [int(j) for j in y[i]]
    return y, z


# def make_hist_map(image, channel):
#     bins = list(range(0, 10000, 10))
#     pix_num = image.shape[0] * image.shape[1]
#     hist = [(np.histogram(image[:, :, i], bins=bins)) for i in range(channel)]
#     print('hist1', hist)
#     hist = np.stack([(list(hist[i])) for i in range(channel)])
#     print(hist.shape)
#     print('hist2', hist)
#     hist = hist / pix_num
#     print('hist normed', hist)
#     # print('hist shape', hist.shape)
#     return hist, bins

def make_hist_map(image, channel):
    bins = list(range(0, 10000, 10))
    pix_num = image.shape[0] * image.shape[1]
    for i in range(channel):
        hist, bin_edges = np.histogram(image[:, :, i], bins=bins)
        hist = hist / pix_num
    hist, bin_edges = [(np.histogram(image[:, :, i], bins=bins)) for i in range(channel)]
    print('hist1', hist)
    hist = np.stack([(list(hist[i])) for i in range(channel)])
    print(hist.shape)
    print('hist2', hist)
    # hist = hist / pix_num
    print('hist normed', hist)
    # print('hist shape', hist.shape)
    return hist, bins



param_dict = {'Vnir': [0, 50000, [0, 1000, 0, 840]], 'Swir': [0, 2000, [0, 500, 0, 256]]}


def plot_hist(hist, bins, class_, state):
    c = plt.imshow(hist, vmin=0, vmax=500,
                   extent=[0, 1000, 840, 0],
                   interpolation='nearest', origin='upper')
    plot_ext(class_, state, c)
    plt.show()


def plot_ext(class_, state, plot):
    plt.colorbar(plot)
    plt.grid()
    plt.xlabel('Pix value')
    plt.ylabel('Channel')
    plt.title(f'histogram class {class_} {state}',
              fontweight="bold")


def save_hist(hist, bins, class_, state):
    c = plt.imshow(hist, vmin=0, vmax=2000,
                   extent=[500, 0, 256, 0],
                   interpolation='nearest', origin='upper')

    plt.colorbar()
    plt.grid()
    plt.xlabel('Pix value(/20)')
    plt.ylabel('Channel')
    plt.title(f'histogram class {class_} {state}',
              fontweight="bold")

    fname = '/Users/ramtahor/Desktop/graphs_10-11/graphs_hist/new_graphs/' + f'2_Swir_histogram_class_{class_}_{state}.png'
    if os.path.isfile(fname):
        fname = '/Users/ramtahor/Desktop/graphs_10-11/graphs_hist/new_graphs/' + f'2_Swir_histogram_class_{class_}_{state}_2.png'
    plt.savefig(fname=fname)


def dif_hist(hist1, hist2):
    dif = abs(hist1 - hist2)
    return dif


def convert_rect_to_dict(_list):  # TODO: Add channels
    l = []
    for rect in _list:
        rect, bins = make_hist_map(rect, 256)
        l.append(rect)
        _list = l
    return _list


def PCA_Spectral(key, band_start, band_stop):
    image_folder, class_, chan = get_folder_and_class(key)
    img, im_data, im_time = open_image(image_folder=image_folder)
    pca = spy.principal_components(img[band_start:band_stop])
    return pca

# def make_hist_dict(rect_dict):
#     out_dict = {}
#     for key in rect_dict:
#         img, im_time, channels, class_ = open_image_in_dict(key)
#         crop_list = cut_crop_ope(rect_dict[key], img, 256)
#         print('crop list 0', crop_list[1].shape)
#         # print(crop_list[0])
#         out_dict[key] = (crop_list)
#     return out_dict, channels, im_time, class_

# # Creates a crop from the rectangle coordinates and put it the rectangle list
# def cut_crop_ope(rect_list, img, channels):
#     for i in range(len(rect_list)):
#         rect_list[i] = (
#             img[rect_list[i][1]:rect_list[i][3] + rect_list[i][1], rect_list[i][0]:rect_list[i][2] + rect_list[i][0],
#             :])
#         rect_list[i], bins = make_hist_map(rect_list[i], channels)
#     return rect_list


# def hist_print(_dict):
#
#     for key in _dict:
#         img, state, chan, class_ = h_utils.open_image_in_dict(key)
#         _dict[key] = h_utils.cut_crop(_dict[key], img)
#         print('dict first', (_dict[key][0].shape))
#         _dict[key] = h_utils.convert_rect_to_dict(_dict[key])
#         print('_dict[key]', _dict[key][0].shape)
#         hist = sum(_dict[key])
#         print(hist.shape)
#         h_utils.save_hist(hist, 10, class_, state)
#         plt.imshow(hist, vmin=0, vmax=5000,
#                    extent=[500, 0, 256, 0],
#                    interpolation='nearest', origin='upper')
#         plt.show()
#         plt.close()
# h_utils.save_(class_, state)

# def hist_print_2(r_dict):
#     rect_dict, channels, im_time, class_ = h_utils.make_crops_sick_healthy(r_dict)
#     for key in rect_dict:
#         print(rect_dict[key][0])
#         avg_list_helthy = h_utils.ope(rect_dict[key][1], channels, np.average)
#         avg_list_helthy = h_utils.inlist_avg(avg_list_helthy, channels)
#         line, color = h_utils.check_line_and_color_sick_healthy(key)
#         h_utils.make_plot_new_format(avg_list_helthy, channels, line='solid', color='green', class_=class_, state='healthy')
#         avg_list_sick = h_utils.ope(rect_dict[key][0], channels, np.average)
#         avg_list_sick = h_utils.inlist_avg(avg_list_sick, channels)
#         line, color = h_utils.check_line_and_color_sick_healthy(key)
#         h_utils.make_plot_new_format(avg_list_sick, channels, line, color, class_, state='sick')
#         plt.show()
