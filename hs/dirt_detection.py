import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv

path_image_1 = '/Users/ramtahor/Desktop/Screen Shot.png'
path_image_2 = '/Users/ramtahor/Downloads/image002.jpg'

'''Gets path to image and kernel size and returns a grayscale image after morphological operators'''

def image_cleaning(path, k_size: tuple):
    image = cv.imread(path)
    kernel = np.ones(k_size, np.uint16)
    image = image + cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thr = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)
    return image, thr


'''returns an array with contours filtered by Perimeter/Area and convexity, dataframe of all contours with labels. '''


def draw_con_on_image(image, contours, epsilon, min_ar):
    df = pd.DataFrame(columns=['Center_mass', 'area', 'label'])

    for i, c in enumerate(contours):
        M = cv.moments(c)  # Calculates moment of inertia
        accuracy = epsilon * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, accuracy, True)

        if cv.isContourConvex(c) or (cv.arcLength(c, True) / (cv.contourArea(c) + 10e-6)) > min_ar:
            conts = cv.drawContours(image, [approx], -1, (255, 0, 0), 2)
            df.loc[i, ['Center_mass', 'area', 'label']] = [
                (int(M["m10"] / (M["m00"] + 10e-6)), int(M["m01"] / (M["m00"] + 10e-6))), cv.contourArea(c), 'dot']
        else:
            conts = cv.drawContours(image, [approx], -1, (0, 255, 0), 2)
            df.loc[i, ['Center_mass', 'area', 'label']] = [
                (int(M["m10"] / (M["m00"] + 10e-6)), int(M["m01"] / (M["m00"] + 10e-6))), cv.contourArea(c), 'dirt']
    return conts, df


def main():
    image, thr = image_cleaning(path_image_2, k_size=(3, 3))
    contours, hierarchy = cv.findContours(thr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    con, df = draw_con_on_image(image, contours, epsilon=0.0001, min_ar=0.45)
    print(df)
    plt.figure('contour')
    plt.imshow(con), plt.show()


    plt.show()


if __name__ == '__main__':
    main()
