from skimage import img_as_float

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
        image = img_as_float(image)
        um_image = codes.UM.unsharped_masking(5, 2, image)
        imageio.imwrite(os.path.join(um_image_out_path, i), um_image)
    print("UM images saved")


def save_clahe_images():
    if not os.path.exists(os.path.join(non_covid_path, 'CLAHE_out')):
        os.makedirs(os.path.join(non_covid_path, 'CLAHE_out'))

    clahe_out_path = os.path.join(non_covid_path, 'CLAHE_out')

    for i in non_covid_files:
        image_path = os.path.join(non_covid_path, i)
        image = cv2.imread(image_path, 0)
        thresh_image, clahe_image = codes.CLAHE_OpenCV.clahe_opencv(image, 2.0, 8)
        imageio.imwrite(os.path.join(clahe_out_path, i), clahe_image)
    print("CLAHE images saved")


def save_hef_images():
    if not os.path.exists(os.path.join(non_covid_path, 'HEF_out')):
        os.makedirs(os.path.join(non_covid_path, 'HEF_out'))

    hef_out_path = os.path.join(non_covid_path, 'HEF_out')

    count = 0
    for i in non_covid_files:
        image_path = os.path.join(non_covid_path, i)
        image = imageio.imread(image_path)
        hef_image = codes.HEF.hef(image)
        imageio.imwrite(os.path.join(hef_out_path, i), hef_image)
        count += 1
        if count % 100 == 0:
            print(str(count) + " images saved")
    print("HEF images saved")


if __name__ == "__main__":
    save_um_images()
    # save_clahe_images()
    # save_hef_images()
