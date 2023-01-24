import os
import cv2
import imageio

import codes.CLAHE_OpenCV

# get directories for test files
TEST_COVID_DIR = "D:\\Pycharm\\New\\X-ray-Image-Enhancement\\data\\IEEE\\Covid19-dataset\\test\\Covid"
TEST_NORMAL_DIR = "D:\\Pycharm\\New\\X-ray-Image-Enhancement\\data\\IEEE\\Covid19-dataset\\test\\Normal"
TEST_VIRAL_PNEUMONIA = "D:\\Pycharm\\New\\X-ray-Image-Enhancement\\data\\IEEE\\Covid19-dataset\\test\\Viral Pneumonia"

# get directories for train files
TRAIN_COVID_DIR = "D:\\Pycharm\\New\\X-ray-Image-Enhancement\\data\\IEEE\\Covid19-dataset\\train\\Covid"
TRAIN_NORMAL_DIR = "D:\\Pycharm\\New\\X-ray-Image-Enhancement\\data\\IEEE\\Covid19-dataset\\train\\Normal"
TRAIN_VIRAL_PNEUMONIA = "D:\\Pycharm\\New\\X-ray-Image-Enhancement\\data\\IEEE\\Covid19-dataset\\train\\Viral Pneumonia"

# check if the directory exists
if not os.path.exists(TEST_COVID_DIR) or not os.path.exists(TEST_NORMAL_DIR) or not os.path.exists(
        TEST_VIRAL_PNEUMONIA):
    exit("Directory does not exist")

# create test output directory in the parent directory
OUTPUT_DIR = "D:\\Pycharm\\New\\X-ray-Image-Enhancement\\data\\IEEE\\Covid19-dataset-enhanced"
OUTPUT_DIR_TEST_COVID = os.path.join(OUTPUT_DIR, "test", "Covid")
OUTPUT_DIR_TEST_NORMAL = os.path.join(OUTPUT_DIR, "test", "Normal")
OUTPUT_DIR_TEST_VIRAL_PNEUMONIA = os.path.join(OUTPUT_DIR, "test", "Viral Pneumonia")

# create train output directory in the parent directory
OUTPUT_DIR_TRAIN_COVID = os.path.join(OUTPUT_DIR, "train", "Covid")
OUTPUT_DIR_TRAIN_NORMAL = os.path.join(OUTPUT_DIR, "train", "Normal")
OUTPUT_DIR_TRAIN_VIRAL_PNEUMONIA = os.path.join(OUTPUT_DIR, "train", "Viral Pneumonia")

# check if all the directories exist or not and create them if they don't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(OUTPUT_DIR_TEST_COVID):
    os.makedirs(OUTPUT_DIR_TEST_COVID)

if not os.path.exists(OUTPUT_DIR_TEST_NORMAL):
    os.makedirs(OUTPUT_DIR_TEST_NORMAL)

if not os.path.exists(OUTPUT_DIR_TEST_VIRAL_PNEUMONIA):
    os.makedirs(OUTPUT_DIR_TEST_VIRAL_PNEUMONIA)

if not os.path.exists(OUTPUT_DIR_TRAIN_COVID):
    os.makedirs(OUTPUT_DIR_TRAIN_COVID)

if not os.path.exists(OUTPUT_DIR_TRAIN_NORMAL):
    os.makedirs(OUTPUT_DIR_TRAIN_NORMAL)

if not os.path.exists(OUTPUT_DIR_TRAIN_VIRAL_PNEUMONIA):
    os.makedirs(OUTPUT_DIR_TRAIN_VIRAL_PNEUMONIA)

# generate enhanced images for test files using CLAHE and save them in the output directory


def generate_clahe_images(file_dir: str, output_dir: str):
    files = os.listdir(file_dir)

    for i in files:
        file_name = os.path.join(file_dir, i)
        image = cv2.imread(file_name, 0)
        thresh_image, clahe_image = codes.CLAHE_OpenCV.clahe_opencv(image, 2.0, 8)
        imageio.imwrite(os.path.join(output_dir, i), clahe_image)
    print("done :" + file_dir)


if __name__ == "__main__":
    generate_clahe_images(TEST_COVID_DIR, OUTPUT_DIR_TEST_COVID)
    generate_clahe_images(TEST_NORMAL_DIR, OUTPUT_DIR_TEST_NORMAL)
    generate_clahe_images(TEST_VIRAL_PNEUMONIA, OUTPUT_DIR_TEST_VIRAL_PNEUMONIA)
    generate_clahe_images(TRAIN_COVID_DIR, OUTPUT_DIR_TRAIN_COVID)
    generate_clahe_images(TRAIN_NORMAL_DIR, OUTPUT_DIR_TRAIN_NORMAL)
    generate_clahe_images(TRAIN_VIRAL_PNEUMONIA, OUTPUT_DIR_TRAIN_VIRAL_PNEUMONIA)
