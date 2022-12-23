import codes.HEF
import codes.UM
import codes.CLAHE_OpenCV
import codes.directory

import os
import imageio.v2 as imageio
import cv2

covid_path = os.path.join(codes.directory.parent_dir, codes.directory.covid_dir)
non_covid_path = os.path.join(codes.directory.parent_dir, codes.directory.non_covid_dir)

list_of_covid_files = os.listdir(covid_path)
list_of_non_covid_files = os.listdir(non_covid_path)
covid_files = [file for file in list_of_covid_files if file.endswith(".jpg")]
print("Number of covid files: " + str(len(covid_files)))

non_covid_files = [file for file in list_of_non_covid_files if file.endswith(".jpg")]
print("Number of non-covid files: " + str(len(non_covid_files)))


def save_um_images():
    if not os.path.exists(os.path.join(non_covid_path, 'UM_out')):
        os.makedirs(os.path.join(non_covid_path, 'UM_out'))

    um_image_out_path = os.path.join(non_covid_path, 'UM_out')

    for i in non_covid_files:
        image_path = os.path.join(non_covid_path, i)
        image = imageio.imread(image_path)
        um_image = codes.UM.unsharped_masking(5, 2, image)
        imageio.imwrite(os.path.join(um_image_out_path, i), um_image)
    print("UM images saved")


if __name__ == "__main__":
    save_um_images()
