# -*- coding: utf-8 -*-
"""
main.py
An example script for extracting data from the h5 file (HBTA dataset).

"""

import h5py
import pandas as pd
import numpy as np

from functions import extract_hdf5
import os
os.chdir='D:/data'

h5_file = 'Rt345Bridge.h5'


# ------------------------------------ #
# EXTRACT DATA - Alternative 1
# ------------------------------------ #

# Extract data directly from the h5 file

# Import
data_h5 = h5py.File(h5_file, 'r')
recordings=['baseline_1','baseline_2','damage_1','damage_2','damage_3','damage_4','damage_5','damage_6']

test_groups1=['test_1','test_2','test_3']
test_groups2=['center_lane','east_lane','west_lane']
recordings=np.array(recordings)
for i in range(0,8):
    for j in range(0,3):
        data=pd.DataFrame()
        data = extract_hdf5(h5_file, recordings,test_groups1[j],test_groups2[j])
        data_df = data[recordings[i]]['data_df']
        data_df.to_excel(recordings[i]+'_'+test_groups1[j]+'.xlsx')
        print(data_df)
