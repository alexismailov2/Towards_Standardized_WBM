####### WILSON-COWAN MODEL

#%% Import libraries
import sys

sys.path.insert(0, "../python/pyWilsonSimulation")

import pyWilsonSimulation

from sklearn.preprocessing import MinMaxScaler
from helper_funcs import *
import matplotlib.pyplot as plt
import scipy.signal as signal
import multiprocessing as mp
import scipy.stats as stats
import scipy.io as sio
import numpy as np
import time
import os

# Defining paths
#root_path = 'C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Spring Sem\\iso_dubai\\ISO\\HCP_DTI_BOLD'
#write_path = "C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Dissertation\\Notebooks\\MyCodes\\results"
root_path = '/Users/alexanderismailov/CLionProjects/untitled/tests/HCP_DTI_BOLD'
write_path = "/Users/alexanderismailov/CLionProjects/untitled/tests/results"

# Defining integration parameters
time_simulated = 510.000 # seconds

# Defining when to start saving data
save_data_start = 150.0 # seconds

# Defining the number of threads to use
number_threads_needed = 48

def toSWIG_VectorDouble(np_array):
    data1D = pyWilsonSimulation.VectorDouble()
    for x, element in enumerate(np_array):
        data1D.append(element)
    return data1D

def toSWIG_VectorVectorDouble(np_array):
    data2D = pyWilsonSimulation.VectorVectorDouble()
    for y, row in enumerate(np_array):
        data2D.append(toSWIG_VectorDouble(row))
    return data2D

def toSWIG_VectorInt(np_array):
    data1D = pyWilsonSimulation.VectorInt()
    for x, element in enumerate(np_array):
        data1D.append(element.astype(int))
    return data1D

def toSWIG_VectorVectorInt(np_array):
    data2D = pyWilsonSimulation.VectorVectorInt()
    for y, row in enumerate(np_array):
        data2D.append(toSWIG_VectorInt(row))
    return data2D

def toSWIG_VectorVectorVectorDouble(np_array):
    data3D = pyWilsonSimulation.VectorVectorVectorDouble()
    for z, row in enumerate(np_array):
        data3D.append(toSWIG_VectorVectorDouble(row))
    return data3D

#%% Start main program
if __name__ == "__main__":

    # %% Initial operations - checking, printing, etc.

    # Printing messages for start and beginning timer
    print('Running Wilson-Cowan model...')
    start_time = time.time()
    
    # Checking root path type
    check_type(root_path, str, 'root_path')

    # Checking integration parameters
    check_type(time_simulated, float, 'time_simulated')

    # Checking number of threads
    check_type(number_threads_needed, int, 'number_threads_needed')


    #%% Load empirical data
    SC_matrix = get_empirical_SC(root_path)
    FC_matrix = get_empirical_FC(root_path)
    BOLD_signals = get_empirical_BOLD(root_path)

    #%% Check number of available threads - multiprocessing tingz

    # Get number of available threads
    number_threads_available = mp.cpu_count()

    # Check if number of threads is greater than available threads
    if number_threads_needed > number_threads_available:
        # If so, set number of threads to available threads
        number_threads_needed = number_threads_available
        # Print message to confirm
        print('Number of threads needed is greater than available threads. Setting number of threads to available threads.')
        print('Number of threads needed: ' + str(number_threads_needed))
        print('Number of threads available: ' + str(number_threads_available))
    else:
        # Otherwise, print message to confirm
        print('Number of threads needed is less than or equal to available threads. Setting number of threads to number of threads needed.')
        print('Number of threads needed: ' + str(number_threads_needed))
        print('Number of threads available: ' + str(number_threads_available))

    #%% Run the simulation and get results
    
    # Define start time before simulation
    start_time = time.time()

    wilsonConfig = pyWilsonSimulation.WillsonConfig()
    wilsonConfig.coupling_strength = 0.0
    wilsonConfig.delay = 0.0

    wilsonConfig.structural_connectivity_mat = toSWIG_VectorVectorDouble(SC_matrix)
    wilsonConfig.wilson_number_of_oscillators = 100
    
    wilsonConfig.wilson_c_ee = 16.0
    wilsonConfig.wilson_c_ei = 12.0
    wilsonConfig.wilson_c_ie = 15.0
    wilsonConfig.wilson_c_ii = 3.0

    # TODO: Not used right now
    coupling_matrix = wilsonConfig.coupling_strength * SC_matrix
    coupling_matrix += (
        np.diag(np.ones((wilsonConfig.wilson_number_of_oscillators,)) * wilsonConfig.wilson_c_ee)
    )

    wilsonConfig.wilson_tau_e = 8.0
    wilsonConfig.wilson_tau_i = 8.0

    wilsonConfig.wilson_r_e = 1.0
    wilsonConfig.wilson_r_i = 1.0

    wilsonConfig.wilson_alpha_e = 1.0
    wilsonConfig.wilson_alpha_i = 1.0
    wilsonConfig.wilson_theta_e = 4.0
    wilsonConfig.wilson_theta_i = 3.7

    wilsonConfig.wilson_external_e = 0.1
    wilsonConfig.wilson_external_i = 0.1

    integration_step_size = 0.002
    wilsonConfig.wilson_number_of_integration_steps = int(time_simulated / integration_step_size)
    # TODO: wilson_integration_step_size may be should be int
    wilsonConfig.wilson_integration_step_size = integration_step_size

    # TODO: start_save_idx used in wilson simulation fiunction but not in the cpp
    downsampling_rate = 400
    start_save_idx = int(save_data_start / integration_step_size) + downsampling_rate


    delay_matrix = wilsonConfig.delay * SC_matrix

    upper_idxs_mat = np.floor(delay_matrix / integration_step_size)
    wilsonConfig.wilson_upper_idxs_mat = toSWIG_VectorVectorDouble(upper_idxs_mat)
    wilsonConfig.wilson_lower_idxs_mat = toSWIG_VectorVectorDouble(upper_idxs_mat + 1)

    wilsonConfig.wilson_e_values = toSWIG_VectorDouble(np.random.rand(wilsonConfig.wilson_number_of_oscillators))
    wilsonConfig.wilson_i_values = toSWIG_VectorDouble(np.random.rand(wilsonConfig.wilson_number_of_oscillators))
    # TODO: can be enum
    wilsonConfig.wilson_noise_type = 1
    wilsonConfig.wilson_noise_amplitude = 0.001

    wilsonConfig.wilson_order = 2
    wilsonConfig.wilson_cutoffLow = 0.01
    wilsonConfig.wilson_cutoffHigh = 0.1
    wilsonConfig.wilson_sampling_rate = 1 / 0.7

    wilsonConfig.emp_BOLD_signals = toSWIG_VectorVectorVectorDouble(BOLD_signals)
    wilsonConfig.num_BOLD_subjects = BOLD_signals.shape[0]
    wilsonConfig.num_BOLD_regions = BOLD_signals.shape[1]
    wilsonConfig.num_BOLD_timepoints = BOLD_signals.shape[2]
    
    wilsonConfig.wilson_BO_n_iter = 50
    wilsonConfig.wilson_BO_n_inner_iter = 10
    wilsonConfig.wilson_BO_iter_relearn = 10
    wilsonConfig.wilson_BO_init_samples = 10
    wilsonConfig.wilson_BO_init_method = 1
    wilsonConfig.wilson_BO_verbose_level = 2
    wilsonConfig.wilson_BO_log_file = 'wilson_bo_log.txt'
    wilsonConfig.wilson_BO_surrogate = "sGaussianProcessML"
    wilsonConfig.wilson_BO_sc_type = 0
    wilsonConfig.wilson_BO_l_type = 3
    wilsonConfig.wilson_BO_l_all = False
    wilsonConfig.wilson_BO_epsilon = 0.01
    wilsonConfig.wilson_BO_force_jump = 0
    wilsonConfig.wilson_BO_crit_name = "cIntegratedEI"

    # TODO: Not used
    # --------- Check the shape of the input arguments
    check_shape(FC_matrix, (wilsonConfig.wilson_number_of_oscillators, wilsonConfig.wilson_number_of_oscillators), 'FC_matrix')

    # Run the simulation
    wilson = pyWilsonSimulation.Wilson(wilsonConfig)
    wilson_results = wilson.process()

    # Check results shape
    # check_shape(wilson_results, (wilsonConfig.wilson_number_of_oscillators, wilsonConfig.wilson_number_of_integration_steps + 1), 'wilson_simulation_results')

    # Define end time after simulation
    end_time = time.time()

    # Print the time taken for the simulation
    print('Time taken for simulation: ' + str(end_time - start_time) + ' seconds')
