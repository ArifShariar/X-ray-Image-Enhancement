import imageio.v2 as imageio

import codes.UM

import time
import csv

import os
import matplotlib.pyplot as plt


def benchmark_um():
    parent_dir = 'D:\\Pycharm\\X-ray-Image-Enhancement\\data\\'
    covid_dir = "COVID-19\\"
    non_covid_dir = "Non-COVID-19\\"

    list_of_covid_files = os.listdir(parent_dir + covid_dir)
    list_of_non_covid_files = os.listdir(parent_dir + non_covid_dir)

    # get only *.jpg files
    covid_files = [file for file in list_of_covid_files if file.endswith(".jpg")]
    print("Number of covid files: " + str(len(covid_files)))
    count = 0

    covid_time_count = {}

    start = time.time()
    for i in covid_files:
        image = imageio.imread(parent_dir + covid_dir + i)
        codes.UM.unsharped_masking(5, 2, image)
        count += 1
        # take time count for every 10 images
        if count % 10 == 0:
            covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for Covid files: " + str(end - start))
    print("Average time for Covid files: " + str((end - start) / len(covid_files)))

    non_covid_files = [file for file in list_of_non_covid_files if file.endswith(".jpg")]
    print("Number of non-covid files: " + str(len(non_covid_files)))

    count = 0

    non_covid_time_count = {}

    start = time.time()
    for i in non_covid_files:
        image = imageio.imread(parent_dir + non_covid_dir + i)
        codes.UM.unsharped_masking(5, 2, image)
        count += 1
        # take time count for every 10 images
        if count % 10 == 0:
            non_covid_time_count[count] = time.time() - start

    end = time.time()
    print("Total time for Non-Covid files: " + str(end - start))
    print("Average time for Non-Covid files: " + str((end - start) / len(non_covid_files)))

    # convert covid_time_count and non_covid_time_count to csv file
    with open('covid_time_count_UM.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in covid_time_count.items():
            writer.writerow([key, value])

    with open('non_covid_time_count_UM.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in non_covid_time_count.items():
            writer.writerow([key, value])

    # plot the graph
    plt.plot(list(covid_time_count.keys()), list(covid_time_count.values()))
    plt.plot(list(non_covid_time_count.keys()), list(non_covid_time_count.values()))
    plt.xlabel("Number of images")
    plt.ylabel("Time taken")
    plt.title("Time taken for Unsharp Masking")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    benchmark_um()
