###################################
# this is the code for HEF - High-frequency Emphasis Filter
# author: Arif Shariar Rahman
# email: 1705095@ugrad.cse.buet.ac.bd
# author: Sadia Saman
# email: 1705102@ugrad.cse.buet.ac.bd
# algorithm of HEF:
# 1. Take an image as input
# 2. Compute the Fast Fourier Transform (FFT) and FFT Shift of the image
# 3. Compute the HFE filter using Gaussian High Pass Filter
# 4. Multiply the FFT of the image with the HFE filter
# 5. Perform inverse Fourier Transform and generate the output image
# After this step, we have to perform histogram equalization on the output image
# algorithm of histogram equalization:
# 1. Calculate Probability density function (PDF) of the image
# 2. Calculate Cumulative distribution function (CDF) of the image
# 3. Calculate the new pixel values using the CDF
###################################

import imageio.v2 as imageio
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from codes.Utils import convert_to_grayscale, normalize_image, histogram


def hef(image_in: np.ndarray) -> np.ndarray:
    # image = imageio.imread('D:\\Pycharm\\X-ray-Image-Enhancement\\data\\010.jpg')

    # convert the image to grayscale
    image_greyscale = convert_to_grayscale(image_in)

    image_in = normalize_image(np.min(image_greyscale), np.max(image_in), 0, 255, image_greyscale)

    # compute the FFT of the image
    image_fft = fft2(image_in)
    image_sfft = fftshift(image_fft)

    m, n = image_sfft.shape
    filter_array = np.zeros((m, n))

    dov = 30
    # compute the HFE filter
    for i in range(m):
        for j in range(n):
            filter_array[i, j] = 1.0 - np.exp(-((i - m / 2.0) ** 2 + (j - n / 2.0) ** 2) / (2 * (dov ** 2)))

    k1 = 0.5
    k2 = 0.75

    high_filter = k1 + k2 * filter_array

    image_filtered = image_sfft * high_filter

    image_hef = np.real(ifft2(fftshift(image_filtered)))

    # histogram equalization
    hist, bins = histogram(image_hef)

    # calculate the probability density function
    pixel_probability = hist / hist.sum()

    # calculate the cumulative distribution function
    cdf = np.cumsum(pixel_probability)
    cdf_normalized = cdf * 255
    hist_eq = {}
    for i in range(len(cdf)):
        hist_eq[bins[i]] = int(cdf_normalized[i])

    for i in range(m):
        for j in range(n):
            image_in[i, j] = hist_eq[image_hef[i, j]]

    return image_in.astype(np.uint8)


if __name__ == '__main__':
    image = imageio.imread('D:\\Pycharm\\X-ray-Image-Enhancement\\data\\010.jpg')
    image = hef(image)
    imageio.imwrite('D:\\Pycharm\\X-ray-Image-Enhancement\\data\\010_hef_new.jpg', image)
