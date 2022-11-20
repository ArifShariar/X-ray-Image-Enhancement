import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
image = cv2.imread("/home/arif/PycharmProjects/X-ray-Image-Enhancement/data/COVID-19/COVID-19 (1).png", 0)
equalized_image = cv2.equalizeHist(image)
plt.title("Original Image")
plt.hist(image.flat, bins=100, range=(0, 255))
plt.show()
plt.title("Equalized Image")
plt.hist(equalized_image.flat, bins=100, range=(0, 255))
plt.show()


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(equalized_image)
plt.title("CLAHE Image")
plt.hist(cl1.flat, bins=100, range=(0, 255))
plt.show()

ret, thresh1 = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.title("Threshold Image")
plt.hist(thresh1.flat, bins=100, range=(0, 255))
plt.show()

# show the image
cv2.imshow("Original Image", image)
cv2.imshow("Equalized Image", equalized_image)
cv2.imshow("CLAHE Image", cl1)
cv2.imshow("Threshold Image", thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

