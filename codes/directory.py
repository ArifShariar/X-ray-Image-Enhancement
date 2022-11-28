############################
#
# This file is for managing the directories
# I was facing problem with the relative path, so I created this python file
# to have all directories in one place
# if someone wants to change the directory, they can do it here
#
############################

import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\"
covid_dir = "data\\COVID-19\\"
non_covid_dir = "data\\Non-COVID-19\\"

csv_dir = "benchmarks\\"


iee_dir = "data\\IEEE\\"
ieee_covid_dir = iee_dir + "Covid19-dataset\\train\\Covid\\"
ieee_non_covid_dir = iee_dir + "Covid19-dataset\\train\\Normal\\"




