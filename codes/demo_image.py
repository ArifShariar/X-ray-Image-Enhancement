import numpy as np
import matplotlib.pyplot as plt

image = open('data/010.jpg', 'rb').read()

#check if the image is open
if image is not None:
    print('Image is open')