import os.path

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
    image_in = os.path.join(codes.directory.parent_dir, 'data', '010.jpg')
    image = cv2.imread(image_in, 0)
    thresh1, cl1 = clahe_opencv(image, 2.0, 8)
    thresh_out = os.path.join(codes.directory.parent_dir, 'data', '010_CLAHE_image_thresh.jpg')
    imageio.imwrite(thresh_out, thresh1)
    cl1_out = os.path.join(codes.directory.parent_dir, 'data', '010_CLAHE_image_cl1.jpg')
    imageio.imwrite(cl1_out, cl1)
