import argparse
import os
import cv2 as cv
import numpy as np
import pandas as pd
import hs.hs_data_utils as hsu

parser = argparse.ArgumentParser(description='Arguments for home assignment selection')
parser.add_argument('--path_to_poly', default='/Users/ramtahor/Desktop/poly-json',
                    help='Path to a polygon files directory')
parser.add_argument('--path_to_images', default='/Volumes/My Passport/Parsimons', help="path to images directory")
parser.add_argument('--directory', default='/Users/ramtahor/Desktop/parsimmon_dataset/',
                    help='Directory where the file would be saved')

path = '/Users/ramtahor/Desktop/hand/_Vnir-Parsimons_class-2_51-60_no-leaves_2020-11-19_12-39-37'
poly_path = '/Users/ramtahor/Desktop/poly-json/_Vnir-Parsimons_class-2_51-60_no-leaves_2020-11-19.json'


# this function gets the name parameters from the path and polygons df
# and creates the name for each fruit
def get_name(name_list, label: str):
    number = str(int(name_list[3].split('-')[0]) - 1 + int(label.split('_')[1]))
    f_class = name_list[2]
    time = name_list[5].split('.json')[0]
    return f_class + '_' + number + '_' + time


# This function arrange the DataFrame with the columns: 'label', 'points', 'name'
# where the name is of the form: class-2_51_2020-11-19 -> class_number_date
def arrange_df(json_path: str, df: pd.DataFrame):
    name_list = json_path.split('_')
    df['name'] = None
    for i, row in df.iterrows():
        row['name'] = get_name(name_list, row['label'])
    return df


def cut_box(points, img):
    cnt = np.array(points).astype(np.int)[np.newaxis]
    x, y, w, h = cv.boundingRect(cnt)
    return img[y - 10: y + h + 10, x - 10: x + w + 10, :]


def save_poly_as_array(poly_df, hs_img, dir):
    for i, row in poly_df.iterrows():
        print(row['name'])
        fruit = cut_box(row['points'], hs_img)
        np.save(dir + row['name'], fruit)


def create(**kwargs):
    for folder in os.listdir(kwargs['img_dir']):
        for image_folder in os.listdir(os.path.join(kwargs['img_dir'], folder)):
            for poly_file in os.listdir(kwargs['poly_dir']):
                if poly_file.split('.json')[0] == image_folder:
                    print('image', image_folder)
                    print('poly', poly_file)
                    time = (poly_file.split('_')[-1]).split('.json')[0]
                    df = arrange_df(poly_file, hsu.json_to_df(os.path.join(kwargs['poly_dir'], poly_file)))
                    img = hsu.open_image(os.path.join(kwargs['img_dir'], folder, image_folder))
                    dir = kwargs['directory'] + time + '/'
                    print('dir', dir)
                    save_poly_as_array(df, img, dir)
                    print('saved')


def main():
    args = parser.parse_args()
    create(img_dir=args.path_to_images, poly_dir=args.path_to_poly, directory=args.directory)


if __name__ == '__main__':
    main()
