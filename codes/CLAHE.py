###################################
# this is the code for HEF - High-frequency Emphasis Filter
# author: Arif Shariar Rahman
# email: 1705095@ugrad.cse.buet.ac.bd
# author: Sadia Saman
# email: 1705102@ugrad.cse.buet.ac.bd
# CLAHE - Contrast Limited Adaptive Histogram Equalization
# CLAHE is a variant of Adaptive Histogram Equalization (AHE) which is used to improve the contrast of images.
# AHE divides the image into small blocks called "tiles". Then it
# applies equalization on each of these blocks independently. So in a small area, histogram would confine to a
# small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting
# is applied. If any histogram bin is above the specified contrast limit, those pixels
# are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization,
# to remove artifacts in tile borders, bi-linear interpolation is applied.
# source : https://docs.opencv.org/3.4/d5/daf/tutorial_py_histogram_equalization.html
# source : https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/
###################################

import Utils as Utils
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from skimage import img_as_float

image = imageio.imread('/home/arif/PycharmProjects/X-ray-Image-Enhancement/data/010.jpg')

if len(image.shape) > 2:
    image = Utils.convert_to_grayscale(image)

normalized_image = Utils.normalize_image(np.min(image), np.max(image), 0, 255, image)
imageio.imwrite('/home/arif/PycharmProjects/X-ray-Image-Enhancement/data/010_normalized.jpg', normalized_image)

window_size = 100
clip_limit = 150
iterations = 5

border = window_size // 2
padded_image = np.pad(normalized_image, border, "reflect")
shape = padded_image.shape
padded_equalized_image = np.zeros(shape).astype(np.uint8)

for i in range(border, shape[0] - border):
    if i % 50 == 0:
        print("iteration: ", i)
    for j in range(border, shape[1] - border):
        region = padded_image[i - border:i + border + 1, j - border:j + border + 1]
        hist, bins = Utils.histogram(region)
        clipped_histogram = Utils.clip_histogram(hist, bins, clip_limit)

        # reduce the value above clip_limit
        for k in range(iterations):
            clipped_histogram = Utils.clip_histogram(clipped_histogram, bins, clip_limit)

        # calculate the cdf
        cdf = Utils.calculate_cdf(clipped_histogram, bins)

        padded_equalized_image[i][j] = cdf[padded_image[i][j]]
