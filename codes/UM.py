###################################
# this is the code for un-sharped mask
# author: Arif Shariar Rahman
# email: 1705095@ugrad.cse.buet.ac.bd
#
# algorithm of un-sharped mask:
# 1. convolve the image with a gaussian filter
# 2. subtract the gaussian filtered image from the original image
# 3. add the result to the original image
###################################

import imageio
import numpy as np

from scipy.ndimage import gaussian_filter
from skimage import img_as_float

radius = 5
amount = 2

image = imageio.imread('/home/arif/PycharmProjects/X-ray-Image-Enhancement/data/010.jpg')
image = img_as_float(image)

blurred = gaussian_filter(image, sigma=radius)
mask = image - blurred
sharpened = image + amount * mask
sharpened = np.clip(sharpened, 0, 1)
sharpened = (sharpened * 255).astype(np.uint8)

imageio.imwrite('/home/arif/PycharmProjects/X-ray-Image-Enhancement/data/010_sharpened.jpg', sharpened)
