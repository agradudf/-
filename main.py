# -*- coding: utf-8 -*-
"""
main.py
An example script for extracting data from the h5 file (HBTA dataset).

"""

import h5py
import pandas as pd

from functions import extract_hdf5
import os
os.chdir='D:/data/345'


# ------------------------------------ #
# HDF5 FILE
# ------------------------------------ #

# HDF5 file
h5_file = 'Rt345Bridge.h5'


# ------------------------------------ #
# EXTRACT DATA - Alternative 1
# ------------------------------------ #

# Extract data directly from the h5 file

# Import
data_h5 = h5py.File(h5_file, 'r')

# Variables
recording = 'baseline_1'
sensor_group = 'acceleration'
test_group = 'test_1'
sensor='A01'
channel = 'y'
record=str(pd.DataFrame(data_h5))
print(record)

# DataFrame of a single channel
data_1_df = pd.DataFrame(data=data_h5[recording][sensor_group][test_group][sensor][channel]).T


# Information and view of data
#print(data_1_df.info())
#print()
#print(data_1_df.head())


# ------------------------------------ #
# EXTRACT DATA - Alternative 2
# ------------------------------------ #

# Extract data using a convenient function

# Variables
recordings = ['baseline_1']
test_groups1=['test_1']
test_groups2=['center_lane']
# Extract data
for test_group1 in test_groups1:
    for test_group2 in test_groups2:
        data = extract_hdf5(h5_file, recordings,test_group1,test_group2)

# DataFrame of all channels (of the first recording)
data_df = data[recordings[0]]['data_df']
print(data_df)

# Information and view of data
#print()
#print(data_df.info())
#print()
#print(data_df.head())
