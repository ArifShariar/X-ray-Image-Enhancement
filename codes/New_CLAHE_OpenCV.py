import cv2
import imageio.v2 as imageio

import codes.directory


def clahe_opencv(image, clip_limit, tile_grid_size):
    equalized_image = cv2.equalizeHist(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl1 = clahe.apply(equalized_image)
    ret, thresh1 = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh1, cl1


if __name__ == "__main__":
    image = cv2.imread(codes.directory.parent_dir + "\\data\\010.jpg", 0)
    thresh1, cl1 = clahe_opencv(image, 2.0, 8)
    imageio.imwrite(codes.directory.parent_dir + "\\data\\010_CLAHE_OpenCV.jpg", thresh1)
    imageio.imwrite(codes.directory.parent_dir + "\\data\\010_CLAHE_OpenCV_2.jpg", cl1)
