import csv
import matplotlib.pyplot as plt
import codes.directory


def plot_graphs():
    # open csv file
    with open(codes.directory.parent_dir + codes.directory.csv_dir + 'covid_time_count_UM.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        um_covid = {}

        for row in reader:
            if row:
                um_covid[int(row[0])] = float(row[1])

    with open(codes.directory.parent_dir + codes.directory.csv_dir + 'covid_time_count_CLAHE.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        clahe_covid = {}

        for row in reader:
            if row:
                clahe_covid[int(row[0])] = float(row[1])

    with open(codes.directory.parent_dir + codes.directory.csv_dir + 'covid_time_count_HEF.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        hef_covid = {}

        for row in reader:
            if row:
                hef_covid[int(row[0])] = float(row[1])

    # plot graphs
    plt.plot(list(um_covid.keys()), list(um_covid.values()), label="UM")
    plt.plot(list(clahe_covid.keys()), list(clahe_covid.values()), label="CLAHE")
    plt.plot(list(hef_covid.keys()), list(hef_covid.values()), label="HEF")
    plt.xlabel('Number of images')
    plt.ylabel('Time taken')
    plt.title('Time taken for Covid files')
    plt.legend()
    plt.grid()
    plt.savefig(codes.directory.parent_dir + codes.directory.csv_dir + 'comparison.png')
    plt.show()


if __name__ == "__main__":
    plot_graphs()
