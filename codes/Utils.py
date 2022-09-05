import numpy as np
from collections import OrderedDict


def histogram(data):
    """
    Calculates histogram of the data
    :param data: data for histogram
    :return: histogram, bins
    """
    pixels, count = np.unique(data, return_counts=True)
    _hist = OrderedDict()

    for _i in range(len(pixels)):
        _hist[pixels[_i]] = count[_i]

    return np.array(list(_hist.values())), np.array(list(_hist.keys()))


def normalize_image(old_minimum_value, old_max_value, new_min_value, new_max_value, value):
    """
    This function normalizes the old values to the new value

    parameter:
        old_minimum_value: minimum value of the old range
        old_max_value: maximum value of the old range
        new_min_value: minimum value of the new range
        new_max_value: maximum value of the new range
        value: the value to be normalized
    :return: normalized value
    """

    ratio = (value - old_minimum_value) / (old_max_value - old_minimum_value)
    return (new_min_value + ratio * (new_max_value - new_min_value)).astype(np.uint8)


def convert_to_grayscale(image_input):
    """
    Convert the image to grayscale
    :param image_input: the image to be converted
    :return: the grayscale image
    """
    red_value = image_input[:, :, 0] * 0.2989
    green_value = image_input[:, :, 1] * 0.5870
    blue_value = image_input[:, :, 2] * 0.1140
    converted_image = red_value + green_value + blue_value
    return converted_image.astype(np.uint8)


def clip_histogram(hist, bins, clip_limit):
    """
    This function clips the histogram
    :param hist: frequencies of each pixel
    :param bins: pixels
    :param clip_limit: limit of the pixel frequencies
    :return: clipped list
    """

    number_bins = len(bins)

    excess = 0

    for i in range(number_bins):
        if hist[i] > clip_limit:
            excess += hist[i] - clip_limit
            hist[i] = clip_limit

    for_each_bin = excess // number_bins
    left_over = excess % number_bins

    hist += for_each_bin

    for i in range(left_over):
        hist[i] += 1

    return hist


def calculate_cdf(hist, bins):
    """
    This function calculates the cumulative distribution function
    :param hist: frequencies of each pixel
    :param bins: pixels
    :return: cdf
    """

    # probability of each pixel
    pixel_probability = hist / hist.sum()

    # cumulative distribution function
    cdf = np.cumsum(pixel_probability)

    cdf_normalized = cdf * 255

    histogram_equalized = {}

    for i in range(len(cdf)):
        histogram_equalized[bins[i]] = int(cdf_normalized[i])

    return histogram_equalized
