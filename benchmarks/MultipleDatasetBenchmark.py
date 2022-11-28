import csv
import time

import matplotlib.pyplot as plt
import os

import imageio.v2 as imageio
import cv2

import codes.directory

import codes.HEF
import codes.UM
import codes.New_CLAHE_OpenCV

covid_path = codes.directory.parent_dir + codes.directory.covid_dir
iee_covid_path = codes.directory.parent_dir + codes.directory.ieee_covid_dir

list_of_covid_files = os.listdir(covid_path)
print("Number of covid files: " + str(len(list_of_covid_files)))
iee_list_of_covid_files = os.listdir(iee_covid_path)
print("Number of iee covid files: " + str(len(iee_list_of_covid_files)))

minimum_number_of_files = min(len(list_of_covid_files), len(os.listdir(iee_covid_path)))
print("Minimum number of files for benchmark: " + str(minimum_number_of_files))

list_of_covid_files = list_of_covid_files[:minimum_number_of_files]
iee_list_of_covid_files = iee_list_of_covid_files[:minimum_number_of_files]


def benchmark_um():
    covid_time_count = {}

    count = 0
    start = time.time()
    for i in list_of_covid_files:
        image = imageio.imread(covid_path + i)
        codes.UM.unsharped_masking(5, 2, image)
        count += 1
        covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for Covid files (UM): " + str(end - start))
    print("Average time for Covid files (UM): " + str((end - start) / len(list_of_covid_files)))

    iee_covid_time_count = {}

    count = 0
    start = time.time()
    for i in iee_list_of_covid_files:
        image = imageio.imread(iee_covid_path + i)
        codes.UM.unsharped_masking(5, 2, image)
        count += 1
        iee_covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for IEE Covid files (UM): " + str(end - start))
    print("Average time for IEE Covid files (UM): " + str((end - start) / len(iee_list_of_covid_files)))

    # convert covid_time_count and iee_covid_time_count to csv file
    with open('multiple-dataset-benchmark\\covid_um.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['count', 'time'])
        for key, value in covid_time_count.items():
            writer.writerow([key, value])

    with open('multiple-dataset-benchmark\\iee_covid_um.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['count', 'time'])
        for key, value in iee_covid_time_count.items():
            writer.writerow([key, value])

    plt.plot(covid_time_count.keys(), covid_time_count.values(), label="Covid Files")
    plt.plot(iee_covid_time_count.keys(), iee_covid_time_count.values(), label="IEE Covid Files")
    plt.xlabel("Number of files")
    plt.ylabel("Time (s)")
    plt.title("UM Benchmark")
    plt.legend()
    plt.grid()
    plt.savefig("multiple-dataset-benchmark\\um_benchmark.png")
    plt.show()


def benchmark_clahe():
    covid_time_count = {}

    count = 0
    start = time.time()
    for i in list_of_covid_files:
        image = cv2.imread(covid_path + i, 0)
        codes.New_CLAHE_OpenCV.clahe_opencv(image, 2.0, 8)
        count += 1
        covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for Covid files (CLAHE): " + str(end - start))
    print("Average time for Covid files (CLAHE): " + str((end - start) / len(list_of_covid_files)))

    iee_covid_time_count = {}

    count = 0

    start = time.time()
    for i in iee_list_of_covid_files:
        image = cv2.imread(iee_covid_path + i, 0)
        codes.New_CLAHE_OpenCV.clahe_opencv(image, 2.0, 8)
        count += 1
        iee_covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for IEE Covid files (CLAHE): " + str(end - start))
    print("Average time for IEE Covid files (CLAHE): " + str((end - start) / len(iee_list_of_covid_files)))

    # convert covid_time_count and iee_covid_time_count to csv file
    with open('multiple-dataset-benchmark\\covid_clahe.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['count', 'time'])
        for key, value in covid_time_count.items():
            writer.writerow([key, value])

    with open('multiple-dataset-benchmark\\iee_covid_clahe.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['count', 'time'])
        for key, value in iee_covid_time_count.items():
            writer.writerow([key, value])

    plt.plot(covid_time_count.keys(), covid_time_count.values(), label="Covid Files")
    plt.plot(iee_covid_time_count.keys(), iee_covid_time_count.values(), label="IEE Covid Files")
    plt.xlabel("Number of files")
    plt.ylabel("Time (s)")
    plt.title("CLAHE Benchmark")
    plt.legend()
    plt.grid()
    plt.savefig("multiple-dataset-benchmark\\clahe_benchmark.png")
    plt.show()


if __name__ == "__main__":
    # benchmark_um()
    benchmark_clahe()
