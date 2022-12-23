###################################
# this is the code for un-sharped mask
# author: Arif Shariar Rahman
# email: 1705095@ugrad.cse.buet.ac.bd
# author: Sadia Saman
# email: 1705102@ugrad.cse.buet.ac.bd
# algorithm of un-sharped mask:
# 1. convolve the image with a gaussian filter
# 2. subtract the gaussian filtered image from the original image
# 3. add the result to the original image
###################################
import os.path

import imageio.v2 as imageio
import numpy as np

from scipy.ndimage import gaussian_filter
from skimage import img_as_float

import codes.directory


def unsharped_masking(radius: int, amount: int, image: np.ndarray) -> np.ndarray:
    blurred = gaussian_filter(image, sigma=radius)
    mask = image - blurred
    sharpened = image + amount * mask
    sharpened = np.clip(sharpened, 0, 1)
    sharpened = (sharpened * 255).astype(np.uint8)
    return sharpened


if __name__ == '__main__':
    image_path = os.path.join(codes.directory.parent_dir, 'data', '010.jpg')
    image_in = imageio.imread(image_path)
    image_in = img_as_float(image_in)
    sharpened_image = unsharped_masking(5, 2, image_in)
    image_out = os.path.join(codes.directory.parent_dir, 'data', '010_UM_image.jpg')
    imageio.imwrite(image_out, sharpened_image)
