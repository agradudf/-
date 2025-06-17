# -*- coding: utf-8 -*-
"""
functions.py
Assembly of functions.

"""
import h5py
import pandas as pd


def extract_hdf5(h5_file, recordings,test_group1,test_group2):
    """Extraction of HDF5 data from the HBTA dataset.

    This function imports a HDF5 file. It returns a dictionary containing all
    sensor data from the specified recordings provided in the input.

    Parameters
    ----------
    h5_file: str
        A str containing the name of the HDF5 file (and location).
    recordings : list
        A list of recordings provided in strings.

    Returns
    -------
    data : dict
        A dictionary containing all sensor data for each recording.
    """

    # ------------------------------------ #
    # PARAMETERS
    # ------------------------------------ #

    # Import
    data_h5 = h5py.File(h5_file, 'r')

    # ACCELERATION DATA
    # List of sensors
    accelerometer_list = ['A01', 'A02', 'A03', 'A04', 'A05','A06', 
                         'A07', 'A08', 'A09', 'A10','A11', 'A12', 'A13',
                         'A14', 'A15','A16', 'A17', 'A18','A19', 'A20',
                         'A21', 'A22', 'A23', 'A24','A25', 'A26', 
                         'A27', 'A28','A29', 'A30']

    strain_gage_list = ['S01', 'S02', 'S03', 'S04','S05', 
                   'S06', 'S07', 'S08','S09', 'S10', 'S11', 
                   'S12','S13', 'S14', 'S15', 'S16','S17', 
                   'S18', 'S19', 'S20']

    accelerometer_MVS = ['AS']
    test_groups1=['test_1','test_2','test_3']
    test_groups2=['center_lane','east_lane','west_lane']


    data = {}

    # ------------------------------------ #
    # DATA
    # ------------------------------------ #

    for recording in recordings:
        

        # Acceleration data
        data_1_df = pd.DataFrame()

            # Add data
        for accelerometer in accelerometer_list:
            data_1_df[accelerometer + '/y'] = pd.DataFrame(data_h5[recording]
                                                            ['acceleration']
                                                            [test_group1]
                                                            [accelerometer]
                                                            ['y']).T
            
            data_1_df[accelerometer + '/z'] = pd.DataFrame(data_h5[recording]
                                                            ['acceleration']
                                                            [test_group1]
                                                            [accelerometer]
                                                            ['z']).T

        # Strain data
        data_2_df = pd.DataFrame()

        # Add data
        for strain_gage in strain_gage_list:
                # Data in x-direction
            data_2_df[strain_gage + '/x'] = pd.DataFrame(data_h5[recording]
                                                         ['strain']
                                                         [test_group2]
                                                         [strain_gage]).T

        # ALL DATA
        data_df = pd.concat([data_1_df, data_2_df], axis=1)

        # Update dictionary
        data.update({recording: {'data_acceleration_df': data_1_df,
                                 'data_strain_df': data_2_df,
                                 'data_df': data_df}})

    return data
