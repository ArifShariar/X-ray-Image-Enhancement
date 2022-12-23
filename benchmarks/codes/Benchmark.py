import csv
import os
import time

import imageio.v2 as imageio
import matplotlib.pyplot as plt

import codes.HEF
import codes.UM
import codes.CLAHE_OpenCV

import cv2

import codes.directory

covid_path = os.path.join(codes.directory.parent_dir, codes.directory.covid_dir)
non_covid_path = codes.directory.parent_dir + codes.directory.non_covid_dir
list_of_covid_files = os.listdir(covid_path)
list_of_non_covid_files = os.listdir(non_covid_path)
covid_files = [file for file in list_of_covid_files if file.endswith(".jpg")]
print("Number of covid files: " + str(len(covid_files)))

non_covid_files = [file for file in list_of_non_covid_files if file.endswith(".jpg")]
print("Number of non-covid files: " + str(len(non_covid_files)))


def benchmark_um():
    count = 0

    covid_time_count = {}

    start = time.time()
    for i in covid_files:
        image = imageio.imread(covid_path + i)
        codes.UM.unsharped_masking(5, 2, image)
        count += 1
        # take time count for every 10 images
        if count % 10 == 0:
            covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for Covid files: " + str(end - start))
    print("Average time for Covid files: " + str((end - start) / len(covid_files)))

    count = 0

    non_covid_time_count = {}

    start = time.time()
    for i in non_covid_files:
        image = imageio.imread(non_covid_path + i)
        codes.UM.unsharped_masking(5, 2, image)
        count += 1
        # take time count for every 10 images
        if count % 10 == 0:
            non_covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for Non-Covid files: " + str(end - start))
    print("Average time for Non-Covid files: " + str((end - start) / len(non_covid_files)))

    # convert covid_time_count and non_covid_time_count to csv file
    with open('../covid_time_count_UM.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in covid_time_count.items():
            writer.writerow([key, value])

    with open('../non_covid_time_count_UM.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in non_covid_time_count.items():
            writer.writerow([key, value])

    # plot the graph
    plt.plot(list(covid_time_count.keys()), list(covid_time_count.values()))
    plt.plot(list(non_covid_time_count.keys()), list(non_covid_time_count.values()))
    plt.xlabel("Number of images")
    plt.ylabel("Time taken")
    plt.title("Time taken for Unsharp Masking")
    plt.legend(["Covid", "Non-Covid"])
    plt.grid()
    plt.savefig(codes.directory.parent_dir + '\\data\\UM_Plot.png')
    plt.show()

    return covid_time_count, non_covid_time_count


def benchmark_hef():
    count = 0

    covid_time_count = {}

    start = time.time()
    for i in covid_files:
        # TODO: change the path to the HEF file, for any reason, it is not working
        image = imageio.imread(covid_path + i)
        codes.HEF.hef(image)
        count += 1
        # take time count for every 10 images
        if count % 10 == 0:
            covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for Covid files: " + str(end - start))
    print("Average time for Covid files: " + str((end - start) / len(covid_files)))

    count = 0

    non_covid_time_count = {}

    start = time.time()
    for i in non_covid_files:
        image = imageio.imread(non_covid_path + i)
        codes.HEF.hef(image)
        count += 1
        # take time count for every 10 images
        if count % 10 == 0:
            non_covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for Non-Covid files: " + str(end - start))
    print("Average time for Non-Covid files: " + str((end - start) / len(non_covid_files)))

    # convert covid_time_count and non_covid_time_count to csv file
    with open('../covid_time_count_HEF.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in covid_time_count.items():
            writer.writerow([key, value])

    with open('../non_covid_time_count_HEF.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in non_covid_time_count.items():
            writer.writerow([key, value])

    # plot the graph
    plt.plot(list(covid_time_count.keys()), list(covid_time_count.values()))
    plt.plot(list(non_covid_time_count.keys()), list(non_covid_time_count.values()))
    plt.xlabel("Number of images")
    plt.ylabel("Time taken")
    plt.title("Time taken for HEF")
    plt.legend(["Covid", "Non-Covid"])
    plt.grid()
    plt.savefig(codes.directory.parent_dir + '//data//HEF_Plot.png')
    plt.show()

    return covid_time_count, non_covid_time_count


def benchmark_clahe():
    count = 0

    covid_time_count = {}

    start = time.time()
    for i in covid_files:
        image = cv2.imread(covid_path + i, 0)
        codes.New_CLAHE_OpenCV.clahe_opencv(image, 2.0, 8)
        count += 1
        # take time count for every 10 images
        if count % 10 == 0:
            covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for Covid files: " + str(end - start))
    print("Average time for Covid files: " + str((end - start) / len(covid_files)))

    count = 0

    non_covid_time_count = {}

    start = time.time()
    for i in non_covid_files:
        image = cv2.imread(non_covid_path + i, 0)
        codes.New_CLAHE_OpenCV.clahe_opencv(image, 2.0, 8)
        count += 1
        # take time count for every 10 images
        if count % 10 == 0:
            non_covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for Non-Covid files: " + str(end - start))
    print("Average time for Non-Covid files: " + str((end - start) / len(non_covid_files)))

    # convert covid_time_count and non_covid_time_count to csv file
    with open('../covid_time_count_CLAHE.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in covid_time_count.items():
            writer.writerow([key, value])

    with open('../non_covid_time_count_CLAHE.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in non_covid_time_count.items():
            writer.writerow([key, value])

    # plot the graph
    plt.plot(list(covid_time_count.keys()), list(covid_time_count.values()))
    plt.plot(list(non_covid_time_count.keys()), list(non_covid_time_count.values()))
    plt.xlabel("Number of images")
    plt.ylabel("Time taken")
    plt.title("Time taken for CLAHE")
    plt.legend(["Covid", "Non-Covid"])
    plt.grid()
    plt.savefig(codes.directory.parent_dir + '\\data\\CLAHE_Plot.png')
    plt.show()

    return covid_time_count, non_covid_time_count


if __name__ == "__main__":
    # benchmark_um()
    benchmark_hef()
    # enchmark_clahe()
