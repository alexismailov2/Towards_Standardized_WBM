import simulations as sim
from helper_funcs import *
import numpy as np

def wilson_electrical_sim(args):
    """"
    This function will simulate the Wilson-Cowan model with electrical coupling
    for a given set of parameters. It will return the simulated electrical activity

    Parameters
    ----------
    args : tuple, input model arguments
        args[0] : float, coupling strength
        args[1] : float, delay
        args[2] : int, number of oscillators
        args[3] : float, c_ee
        args[4] : float, c_ei
        args[5] : float, c_ie
        args[6] : float, c_ii
        args[7] : float, tau_e
        args[8] : float, tau_i
        args[9] : float, r_e
        args[10] : float, r_i
        args[11] : float, alpha_e
        args[12] : float, alpha_i
        args[13] : float, theta_e
        args[14] : float, theta_i
        args[15] : float, external_e
        args[16] : float, external_i
        args[17] : int, number of integration steps
        args[18] : float, integration step size
        args[19] : int, start_save_idx
        args[20] : int, downsampling_rate
        args[21] : array, initial conditions_e
        args[22] : array, initial conditions_i
        args[23] : array, SC matrix
        args[24] : array, FC matrix
        args[25] : int, noise type
        args[26] : float, noise amplitude
        args[27] : string, write path

    Returns
    -------
    elec: array, simulated electrical activity

    Equation
    --------
    tau_e * dE/dt = -E + (1 - r_e*E)*S_e(c_ee*E + c_ei*I + external_e)
    tau_i * dI/dt = -I + (1 - r_i*I)*S_I(c_ie*E + c_ii*I + external_i)

    where S_e and S_i are sigmoid functions
        S_e = 1 / (1 + exp(-alpha_e * (x - theta_e)))
        S_i = 1 / (1 + exp(-alpha_i * (x - theta_i)))
    """

    print('------------ In wilson_electrical_sim ------------')
    # --------- Check length of the input arguments
    num_params_expected = 28

    if len(args) != num_params_expected:
        exception_msg = 'Exception in WC model. Expected {} arguments, got {}'.format(num_params_expected, str(len(args)))
        raise Exception(exception_msg)
    
    # --------- Unpack the arguments
    print('-- Unpacking arguments --')
    coupling_strength = args[0]
    delay = args[1]
    number_oscillators = args[2]
    c_ee = args[3]
    c_ei = args[4]
    c_ie = args[5]
    c_ii = args[6]
    tau_e = args[7]
    tau_i = args[8]
    r_e = args[9]
    r_i = args[10]
    alpha_e = args[11]
    alpha_i = args[12]
    theta_e = args[13]
    theta_i = args[14]
    external_e = args[15]
    external_i = args[16]
    number_integration_steps = args[17]
    integration_step_size = args[18]
    start_save_idx = args[19]
    downsampling_rate = args[20]
    initial_cond_e = args[21]
    initial_cond_i = args[22]
    SC = args[23]
    FC = args[24]
    noise_type = args[25]
    noise_amplitude = args[26]
    write_path = args[27]


    # --------- Check the type of the input arguments
    print('-- Checking types --')
    check_type(coupling_strength, float, 'coupling_strength')
    check_type(delay, float, 'delay')
    check_type(number_oscillators, int, 'number_oscillators')
    check_type(c_ee, float, 'c_ee')
    check_type(c_ei, float, 'c_ei')
    check_type(c_ie, float, 'c_ie')
    check_type(c_ii, float, 'c_ii')
    check_type(tau_e, float, 'tau_e')
    check_type(tau_i, float, 'tau_i')
    check_type(r_e, float, 'r_e')
    check_type(r_i, float, 'r_i')
    check_type(alpha_e, float, 'alpha_e')
    check_type(alpha_i, float, 'alpha_i')
    check_type(theta_e, float, 'theta_e')
    check_type(theta_i, float, 'theta_i')
    check_type(external_e, float, 'external_e')
    check_type(external_i, float, 'external_i')
    check_type(number_integration_steps, int, 'number_integration_steps')
    check_type(integration_step_size, float, 'integration_step_size')
    check_type(start_save_idx, int, 'start_save_idx')
    check_type(downsampling_rate, int, 'downsampling_rate')
    check_type(initial_cond_e, np.ndarray, 'initial_cond_e')
    check_type(initial_cond_i, np.ndarray, 'initial_cond_i')
    check_type(SC, np.ndarray, 'SC')
    check_type(FC, np.ndarray, 'FC')
    check_type(noise_type, int, 'noise_type')
    check_type(noise_amplitude, float, 'noise_amplitude')
    check_type(write_path, str, 'write_path')

    # --------- Check the type of data in the input arguments
    check_type(initial_cond_e[0], np.float64, 'initial_cond_e[0]')
    check_type(initial_cond_i[0], np.float64, 'initial_cond_i[0]')
    check_type(SC[0, 0], np.float64, 'SC[0, 0]')
    check_type(FC[0, 0], np.float64, 'FC[0, 0]')

    # --------- Check the shape of the input arguments
    check_shape(initial_cond_e, (number_oscillators,), 'initial_cond_e')
    check_shape(initial_cond_i, (number_oscillators,), 'initial_cond_i')
    check_shape(SC, (number_oscillators, number_oscillators), 'SC')
    check_shape(FC, (number_oscillators, number_oscillators), 'FC')


    # --------- Define initial values to be used in equation, COUPLING AND DELAY
    # COUPLING is either c_ee, if local coupling, or SC, if global coupling
    # DELAY is either 0, if local coupling, or delay * path lengths, if global coupling
    print('-- Defining initial values --')
    coupling_matrix = coupling_strength * SC
    # np.fill_diagonal(coupling_matrix, c_ee)
    coupling_matrix += (
        np.diag(np.ones((number_oscillators,)) * c_ee)
    )
    delay_matrix = delay * SC

    # --------- Define the index matrices for integration (WHAT IS THIS)
    print('-- Defining index matrices --')
    upper_idx = np.floor(delay_matrix / integration_step_size).astype(int)
    lower_idx = upper_idx + 1
    
    print('------------ Before simulation ------------')

    # --------- SIMULATION TIME BABEY
    simulation_results = sim.wilson_model(
        coupling_matrix,
        delay_matrix,
        number_oscillators,
        c_ee,
        c_ei,
        c_ie,
        c_ii,
        tau_e,
        tau_i,
        r_e,
        r_i,
        alpha_e,
        alpha_i,
        theta_e,
        theta_i,
        external_e,
        external_i,
        number_integration_steps,
        integration_step_size,
        lower_idx,
        upper_idx,
        initial_cond_e,
        initial_cond_i,
        noise_type,
        noise_amplitude
    )

    print('------------ After simulation ------------')
    # Check results shape
    check_shape(simulation_results, (number_oscillators, number_integration_steps + 1), 'wilson_simulation_results')

    # --------- Convert electrical to BOLD
    sim_bold = sim.electrical_to_bold(simulation_results, 
                                          number_oscillators,
                                          number_integration_steps,
                                          integration_step_size)
    # Check results shape
    check_shape(sim_bold, (number_oscillators, number_integration_steps + 1), 'wilson_simulation_bold')

    # --------- Ignore initialization (and downsample?)
    sim_bold = sim_bold[:, start_save_idx - downsampling_rate + 1 :]
    # sim_bold = sim_bold[:, start_save_idx:]

    # --------- Determine order parameter
    R_mean, R_std = determine_order_R(simulation_results, number_oscillators, int(1 / integration_step_size))

    # --------- Calculate FC
    sim_bold = process_BOLD(sim_bold)
    sim_FC = np.corrcoef(sim_bold)
    np.fill_diagonal(sim_FC, 0.0)

    # Check the same of the simulated FC matrix
    check_shape(sim_FC, (number_oscillators, number_oscillators), 'sim_FC')
    
    # --------- Calculate simFC <-> empFC correlation
    empFC_simFC_corr = determine_similarity(FC, sim_FC)

    # --------- Save the results
    # Define main folder path
    folder_name = "wilson_Coupling{:.4f}Delay{:.4f}\\".format(coupling_strength, delay)
    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    # Define paths
    electric_path = os.path.join(write_path, os.path.join(folder_name, "electrical.csv"))
    bold_path = os.path.join(write_path, os.path.join(folder_name, "bold.csv"))
    FC_path = os.path.join(write_path, os.path.join(folder_name, "simFC.csv"))
    R_path = os.path.join(write_path, os.path.join(folder_name, "R.txt"))
    empFC_simFC_corr_path = os.path.join(write_path, os.path.join(folder_name, "empFC_simFC_corr.txt"))
    # Downsample BOLD
    sim_bold = sim_bold[:, downsampling_rate - 1 :: downsampling_rate]
    # Save the results
    np.savetxt(electric_path, simulation_results, delimiter=",")
    np.savetxt(bold_path, sim_bold, fmt="% .4f", delimiter=",")
    np.savetxt(FC_path, sim_FC, fmt="% .8f", delimiter=",")
    np.savetxt(R_path, np.array([R_mean, R_std]), delimiter=",")
    np.savetxt(empFC_simFC_corr_path, np.array([empFC_simFC_corr]), delimiter=",")
    
    # --------- Return the results
    # Create dictionary of results
    results = {
        'coupling_strength': coupling_strength,
        'delay': delay,
        'R_mean': R_mean,
        'R_std': R_std,
        'empFC_simFC_corr': empFC_simFC_corr
    }

    return results
