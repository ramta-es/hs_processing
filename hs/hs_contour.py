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

path_image_1 = '/Users/ramtahor/Desktop/Screen Shot.png'
path_image_2 = '/Users/ramtahor/Downloads/image002.jpg'
# sharpen the image


params = cv.SimpleBlobDetector_Params()
print((params))

params.filterByConvexity = True
params.minConvexity = 0.00000001
params.maxConvexity = 500000000000

params.filterByInertia = True
params.minInertiaRatio = 0.000001
# params.maxInertiaRatio = 0.5

params.filterByArea = True
params.minArea = 1

params.filterByCircularity = True
params.minCircularity = 0.75

params.minThreshold = 10



kernel = np.array([[0, -1, 0],
                   [-1, 8, -1],
                   [0, -1, 0]])
kernel = np.ones((3,3),np.uint8)
sd_kernel = np.std(np.ones(3))
mean_kernel = np.mean(np.ones(3))
# image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

image = cv.imread(path_image_2)
opening = image + cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
plt.figure()
plt.imshow(opening)
closing = image + cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)
plt.figure()
plt.imshow(closing)
plt.show()

img = cv.blur(image, ksize=(5, 5))
# img = image
plt.imshow(img), plt.show()
img = cv.filter2D(src=img, ddepth=-1, kernel=kernel)

plt.imshow(img), plt.show()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# thr = cv.adaptiveThreshold((gray), 100, cv.THRESH_BINARY, 5, 2)
ret, thr = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
plt.imshow(thr), plt.show()

detector = cv.SimpleBlobDetector_create(params)
keypoints = detector.detect(thr)
# df = pd.DataFrame(columns=['location', 'size', 'label'])
# df.loc[:, ['location']] = [keypoints[i].pt for i in range(len(keypoints))]


im_with_keypoints = cv.drawKeypoints(image, keypoints, np.array([]), (255, 0, 0),
                                     cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
params.minCircularity = 0.00001
params.maxCircularity = 0.6
detector = cv.SimpleBlobDetector_create(params)
keypoints = detector.detect(thr)



im_with_keypoints2 = cv.drawKeypoints(im_with_keypoints, keypoints, np.array([]), (0, 255, 0),
                                     cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(im_with_keypoints2)

plt.show()


