#include <WilsonSimulation/API.hpp>

#include "simulation_helpers.hpp"

// TODO: Only one function used reduce huge dependency
//#include <gsl/gsl_statistics.h>

#include <boost/throw_exception.hpp>
#include <bayesopt/bayesopt.hpp>
#include <bayesopt/bayesopt.h>

#include <cmath>

namespace {

double gsl_stats_mean(const double data[], const size_t stride, const size_t size)
{
  /* Compute the arithmetic mean of a dataset using the recurrence relation
     mean_(n) = mean(n-1) + (data[n] - mean(n-1))/(n+1)   */

  long double mean = 0;
  for (auto i = 0; i < size; i++)
  {
    mean += (data[i * stride] - mean) / (i + 1);
  }
  return mean;
}

/**
 * Saving bold to file.
 * @param bold
 * @param filePath
 */
void saveBoldSignalToFile(std::vector<std::vector<double>> const& bold,
                          std::string const& filePath)
{
  std::ofstream myfile(filePath);
  size_t BOLD_dims[2] = { bold.size(), bold[0].size() };
  for (size_t i = 0; i < BOLD_dims[0]; ++i)
  {
    for (size_t j = 0; j < BOLD_dims[1]; ++j)
      if (j < (BOLD_dims[1] - 1)) {
        myfile << bold[i][j] << ",";
      }
      else if (j == (BOLD_dims[1] - 1)) {
        myfile << bold[i][j] << "\n";
      }
  }
}

/**
 * Saving emp bold to file.
 * @param empBold
 * @param filePath
 */
void saveEmpBoldToFile(std::vector<std::vector<std::vector<double>>> const& empBold,
                       std::string const& filePath)
{
  std::ofstream myfile;
  myfile.open("temp_arrays/unpacked_emp_BOLD.csv");

  for (size_t i = 0; i < empBold.size(); ++i)
  {
    for (size_t j = 0; j < empBold[0].size(); ++j)
    {
      for (size_t k = 0; k < (empBold[0][0].size() - 1); ++k) {
        myfile << empBold[i][j][k] << ",";
      }
      myfile << empBold[i][j][empBold[0][0].size() - 1] << "\n";
    }
  }
}

/**
 * Given a value x, this function returns the value of the sigmoid function with parameters alpha and theta.
 * @param x input value
 * @param alpha sigmoid parameter (steepness)
 * @param theta sigmoid parameter (inflection point position)
 * @return sigmoid function value at x
 */
double wilson_response_function(double x, double alpha, double theta)
{
  double S = 1 / (1 + std::exp(-alpha * (x - theta)));
  S -= 1 / (1 + exp(alpha * theta));
  return S;
}

} // anonymous

/**
 * This is a function that, given electrical activity, will convert it to BOLD signals.
 * It does so using the Balloon-Windkessel model. Again, the differential equation follows
 * Heun's Method.
 *
 * @param wilson_electrical_activity
 * @param wilson_number_of_oscillators
 * @param wilson_number_of_integration_steps
 * @param wilson_integration_step_size
 * @return BOLD signal of each node
 */
std::vector<std::vector<double>> Wilson::electrical_to_bold(std::vector<std::vector<double>>& wilson_electrical_activity,
                                                            int wilson_number_of_oscillators,
                                                            int wilson_number_of_integration_steps,
                                                            float wilson_integration_step_size)
{
  struct WilsonParams {
    double f;
    double q;
    double s;
    double v;
    double differential_f;
    double differential_q;
    double differential_s;
    double differential_v;
    double differential_f2;
    double differential_q2;
    double differential_s2;
    double differential_v2;
    double activity_f;
    double activity_q;
    double activity_s;
    double activity_v;
    double alpha;
    double gamma;
    double kappa;
    double rho;
    double tau;
    double c1;
    double c3;
  };
  auto wp = std::vector<WilsonParams>(wilson_number_of_oscillators);

  std::default_random_engine generator(1);

  if (wilson_electrical_activity.size() != wilson_number_of_oscillators ||
      wilson_electrical_activity[0].size() != wilson_number_of_integration_steps + 1)
  {
    return {};
  }

  // ------------- Initialize values of state variables, [0, 0.1]
  for (auto& item : wp)
  {
    // Initialize state variables
    item.f = rand_std_uniform(generator) * 0.1;
    item.q = rand_std_uniform(generator) * 0.1;
    item.s = rand_std_uniform(generator) * 0.1;
    item.v = rand_std_uniform(generator) * 0.1;
  }

  // ------------- Initialize values of helper variables
  auto c2 = 2.000;
  auto V0 = 0.020;
  for (auto& item : wp)
  {
    // Initialize helper variables
    item.alpha = 1 / (0.320 + rand_std_normal(generator) * 0.039);
    item.gamma = 0.410 + rand_std_normal(generator) * 0.045;
    item.kappa = 0.650 + rand_std_normal(generator) * 0.122;
    item.rho = 0.340 + rand_std_normal(generator) * 0.049;
    item.tau = 0.980 + rand_std_normal(generator) * 0.238;
    item.c1 = 7.0 * item.rho;
    item.c3 = 2.0 * item.rho - 0.2;
  }

  // ------------- Declare output variables
  auto output_bold = std::vector<double>(wilson_number_of_oscillators);
  auto output_bold_matrix = std::vector<std::vector<double>>(wilson_number_of_oscillators, std::vector<double>(wilson_number_of_integration_steps + 1, 0));

  // ------------- Initialize output matrix
  // TODO: should be used std::transform instead
  for (int i = 0; i < wilson_number_of_oscillators; i++)
  {
    output_bold[i] = wp[i].c1 * (1 - wp[i].q);
    output_bold[i] += c2 * (1 - wp[i].q / wp[i].v);
    output_bold[i] += wp[i].c3 * (1 - wp[i].v);
    output_bold[i] *= V0;
  }

  std::vector<double> input_e(wilson_number_of_oscillators);

  // ------------- CONVERSIONS BABEY
  for (int step = 1; step <= wilson_number_of_integration_steps; step++)
  {
    for (int i = 0; i < wilson_number_of_oscillators; i++)
    {
      input_e[i] = wilson_electrical_activity[i][step];
    }

    // ------------ Heun's Method - Step 1
    int i = 0;
    for(auto& item : wp)
    {
      // Calculate differentials
      item.differential_f = item.s;
      item.differential_q = 1 - pow(1 - item.rho, 1 / item.f);
      item.differential_q *= item.f / item.rho;
      item.differential_q -= item.q * pow(item.v, item.alpha - 1);
      item.differential_q /= item.tau;
      item.differential_s = input_e[i++];
      item.differential_s -= item.kappa * item.s + item.gamma * (item.f - 1);
      item.differential_v = (item.f - pow(item.v, item.alpha)) / item.tau;

      // First estimate of the new activity values
      item.activity_f = item.f + wilson_integration_step_size * item.differential_f;
      item.activity_q = item.q + wilson_integration_step_size * item.differential_q;
      item.activity_s = item.s + wilson_integration_step_size * item.differential_s;
      item.activity_v = item.v + wilson_integration_step_size * item.differential_v;
    }

    // ------------ Heun's Method - Step 2
    int j = 0;
    for(auto& item : wp)
    {
      // Calculate differentials
      item.differential_f2 = item.activity_s;
      item.differential_q2 = 1 - pow(1 - item.rho, 1 / item.activity_f);
      item.differential_q2 *= item.activity_f / item.rho;
      item.differential_q2 -= item.activity_q * pow(item.activity_v, item.alpha - 1);
      item.differential_q2 /= item.tau;
      item.differential_s2 = input_e[j++];
      item.differential_s2 -= item.kappa * item.activity_s + item.gamma * (item.activity_f - 1);
      item.differential_v2 = (item.activity_f - pow(item.activity_v, item.alpha)) / item.tau;

      // Second estimate of the new activity values
      item.f += wilson_integration_step_size / 2 * (item.differential_f + item.differential_f2);
      item.q += wilson_integration_step_size / 2 * (item.differential_q + item.differential_q2);
      item.s += wilson_integration_step_size / 2 * (item.differential_s + item.differential_s2);
      item.v += wilson_integration_step_size / 2 * (item.differential_v + item.differential_v2);
    }

    // Calculate BOLD signal
    int osc = 0;
    for (auto const& item : wp)
    {
      output_bold[osc] = item.c1 * (1 - item.q);
      output_bold[osc] += c2 * (1 - item.q / item.v);
      output_bold[osc] += item.c3 * (1 - item.v);
      output_bold[osc] *= V0;
      output_bold_matrix[osc][step] = output_bold[osc];
      osc++;
    }
  }

  // ------------- Unpack the BOLD signal
  printf("----------- Unpacking BOLD signal -----------\n");
  // Printing the dimensions of the BOLD_array
  size_t BOLD_dims[2] = { output_bold_matrix.size(), output_bold_matrix[0].size() };

  auto& unpack_bold = output_bold_matrix;
#if 0 // TODO: Not needed since output_bold_matrix it is the same
  // Create a vector of doubles, to store the entire BOLD signal
  std::vector<std::vector<double>> unpack_bold;
  for (int i = 0; i < BOLD_dims[0]; ++i)
  {
    printf("In processing signal %d\n", i);

    // Create another vector of doubles, to store timesamples for each BOLD signal
    std::vector<double> samples_array;

    // For each BOLD signal in the BOLD signals, for each timestep
    for (int j = 0; j < BOLD_dims[1]; ++j)
    {
      // if (j % 10000 == 0)
      //     printf("In timesample %d\n", j);

      // This will store the value in the bold array
      double value;

      // Get the time_sample point
      PyObject* time_sample = PyArray_GETITEM(BOLD_array, PyArray_GETPTR2(BOLD_array, i, j));

      // Check thet each time sample is a float
      if(PyFloat_Check(time_sample))
        value = PyFloat_AsDouble(time_sample);
      else {
        printf("Not floats!!!");
        PyErr_SetString(PyExc_TypeError, "must pass in list of list of number");
        return {};
      }
      samples_array.push_back(value);
      // Decrement the pointer reference
      Py_DECREF(time_sample);
    }
    unpack_bold.push_back(samples_array);
  }
#endif

  return unpack_bold;
}

Wilson::Wilson(WillsonConfig config)
    : config(std::move(config))
    , wilson_electrical_activity{
        std::vector<std::vector<double>>(config.wilson_number_of_oscillators,
                                         std::vector<double>(config.wilson_number_of_integration_steps + 1, nan("")))}
{
}

std::vector<std::vector<double>> Wilson::process()
{
  if (!config.valid())
  {
    throw std::runtime_error("Not valid config");
  }
  printf("----------------- In CPP file for Wilson Function -----------------\n");

  // TODO: I do not know if it can wilson_electrical_activity used directly
  wilson_output_e = wilson_electrical_activity;
  // ------------- Convert input variables to C++ types
  printf("---- Converting input variables to C++ types ----\n");
  for (int i = 0; i < config.wilson_number_of_oscillators; i++)
  {
    // ------------ Initialize output matrix
    wilson_output_e[i][0] = config.wilson_e_values[i];
    // Other values in matrix are NaN
  }

  // ------------ Get the BOLD signals for processing
  printf("---- Get the empirical BOLD signals for processing ----\n");
//  npy_intp emp_BOLD_dims[] = {PyArray_NDIM(BOLD_signals)};
//  emp_BOLD_dims[0] = PyArray_DIM(BOLD_signals, 0);
//  emp_BOLD_dims[1] = PyArray_DIM(BOLD_signals, 1);
//  emp_BOLD_dims[2] = PyArray_DIM(BOLD_signals, 2);
  size_t emp_BOLD_dims[3] = { config.emp_BOLD_signals.size(),
                              config.emp_BOLD_signals[0].size(),
                              config.emp_BOLD_signals[0][0].size() };
  auto& unpack_emp_BOLD = config.emp_BOLD_signals;
#if 0 // TODO: not needed anymore
  // Create a vector of vectors of vectors for the BOLD signals for all subjects
  std::vector<std::vector<std::vector<double>>> unpack_emp_BOLD;
  // For each subject
  for (int subject = 0; subject < emp_BOLD_dims[0]; ++subject)
  {
    printf("In processing subject %d\n", subject);
    // Create another vector of vector of doubles, to store each subject's 100 region signals
    std::vector<std::vector<double>> subject_regions;

    // For each BOLD signal in the BOLD signals, for each timestep
    for (int region = 0; region < emp_BOLD_dims[1]; ++region)
    {
      if (region % 10 == 0)
        printf("In region %d\n", region);

      // Create a last vector of doubles, to store the timesamples for each signal
      std::vector<double> region_timesamples;

      for (int timepoint = 0; timepoint < emp_BOLD_dims[2]; ++timepoint)
      {
        // This will store the value in the bold array
        double value;

        // Get the time_sample point
        PyObject* time_sample = PyArray_GETITEM(emp_BOLD_signals, PyArray_GETPTR3(emp_BOLD_signals, subject, region, timepoint));

        // Check thet each time sample is a float
        if(PyFloat_Check(time_sample))
          value = PyFloat_AsDouble(time_sample);
        else {
          printf("Not floats!!!");
          PyErr_SetString(PyExc_TypeError, "Empirical BOLD is not in the correct format");
          return {};
        }

        region_timesamples.push_back(value);
        // Decrement the pointer reference
        Py_DECREF(time_sample);
      }
      subject_regions.push_back(region_timesamples);
    }
    unpack_emp_BOLD.push_back(subject_regions);
  }
#endif

  // Saving it just for a sanity check
  printf("----------- Saving unpacked empirical BOLD signal -----------\n");
  saveEmpBoldToFile(unpack_emp_BOLD, "temp_arrays/unpacked_emp_BOLD.csv");
#if 0
  std::ofstream myfile;
  myfile.open("temp_arrays/unpacked_emp_BOLD.csv");

  for (size_t i = 0; i < emp_BOLD_dims[0]; ++i)
  {
    for (size_t j = 0; j < emp_BOLD_dims[1]; ++j)
    {
      for (size_t k = 0; k < emp_BOLD_dims[2]; ++k)
        if (k < (emp_BOLD_dims[2] - 1)) {
          myfile << unpack_emp_BOLD[i][j][k] << ",";
        }
        else if (k == (emp_BOLD_dims[2] - 1)) {
          myfile << unpack_emp_BOLD[i][j][k] << "\n";
        }
    }
  }
#endif

  printf("----------- Filtering the empirical BOLD signal -----------\n");
  // Create a vector that stores for ALL SUBJECTS
  std::vector<std::vector<std::vector<double>>> emp_bold_filtered;

  // For each subject
  for (int subject = 0; subject < emp_BOLD_dims[0]; ++subject)
  {
    printf("In filtering subject %d\n", subject);

    // Add the subject to the vector of all subjects
    emp_bold_filtered.emplace_back(process_BOLD(unpack_emp_BOLD[subject],
                                                emp_BOLD_dims[1],
                                                emp_BOLD_dims[2],
                                                config.wilson_order,
                                                config.wilson_cutoffLow,
                                                config.wilson_cutoffHigh,
                                                config.wilson_sampling_rate));
  }

  // Saving it just for a sanity check
  printf("----------- Saving filtered empirical BOLD signal -----------\n");
  saveEmpBoldToFile(emp_bold_filtered, "temp_arrays/filtered_emp_BOLD.csv");
#if 0
  std::ofstream myfile2;
  myfile2.open("temp_arrays/filtered_emp_BOLD.csv");

  for (size_t i = 0; i < emp_BOLD_dims[0]; ++i)
  {
    for (size_t j = 0; j < emp_BOLD_dims[1]; ++j)
    {
      for (size_t k = 0; k < emp_BOLD_dims[2]; ++k)
        if (k < (emp_BOLD_dims[2] - 1)) {
          myfile2 << emp_bold_filtered[i][j][k] << ",";
        }
        else if (k == (emp_BOLD_dims[2] - 1)) {
          myfile2 << emp_bold_filtered[i][j][k] << "\n";
        }
    }
  }
#endif

  // ------------- Getting the empirical FC
  printf("----------- Getting the empirical FC -----------\n");
  // Create a vector of vectors of vectors for the FC for all subjects
  std::vector<std::vector<std::vector<double>>> unpack_emp_FC;

  // For each subject
  for (int subject = 0; subject < emp_BOLD_dims[0]; subject++)
  {
    // Add the subject to the vector of all subjects
    printf("subject: %d\n\r", subject);
    unpack_emp_FC.emplace_back(determine_FC(emp_bold_filtered[subject]));
  }

  // Saving it just for a sanity check
  printf("----------- Saving unpacked empirical FC -----------\n");
  saveEmpBoldToFile(unpack_emp_FC, "temp_arrays/emp_FC_all.csv");
#if 0
  std::ofstream myfile3;
  myfile3.open("temp_arrays/emp_FC_all.csv");

  for (size_t i = 0; i < emp_BOLD_dims[0]; ++i)
  {
    for (size_t j = 0; j < emp_BOLD_dims[1]; ++j)
    {
      for (size_t k = 0; k < emp_BOLD_dims[1]; ++k)
        if (k < (emp_BOLD_dims[1] - 1)) {
          myfile3 << unpack_emp_FC[i][j][k] << ",";
        }
        else if (k == (emp_BOLD_dims[1] - 1)) {
          myfile3 << unpack_emp_FC[i][j][k] << "\n";
        }
    }
  }
#endif

  // ------------- Finding the average across subjects
  printf("----------- Finding the average across subjects -----------\n");
  // Note that this average FC is what's gonna be stored in the empFC global variable

  std::vector<std::vector<double>> emp_FC;

  // For each region
  for (int i = 0; i < emp_BOLD_dims[1]; i++)
  {
    // Create a vector of doubles for each *other* region
    std::vector<double> region_avg;

    // For each other region
    for (int j = 0; j < emp_BOLD_dims[1]; j++)
    {
      // Create a vector of doubles for each subject
      std::vector<double> subject_values;

      // For each subject
      for (int k = 0; k < emp_BOLD_dims[0]; k++)
      {
        subject_values.push_back(unpack_emp_FC[i][j][k]);
      }
      // Get the mean of the subject values
      double mean = gsl_stats_mean(subject_values.data(), 1, subject_values.size());
      region_avg.push_back(mean);
    }
    emp_FC.push_back(region_avg);
  }

  // Saving it just for a sanity check
  printf("----------- Saving average empirical FC -----------\n");
  saveBoldSignalToFile(emp_FC, "temp_arrays/empFC.csv");
#if 0
  std::ofstream myfile4;
  myfile4.open("temp_arrays/empFC.csv");

  for (size_t i = 0; i < emp_BOLD_dims[1]; ++i)
  {
    for (size_t j = 0; j < emp_BOLD_dims[1]; ++j)
      if (j < (emp_BOLD_dims[1] - 1)) {
        myfile4 << ::emp_FC[i][j] << ",";
      }
      else if (j == (emp_BOLD_dims[1] - 1)) {
        myfile4 << ::emp_FC[i][j] << "\n";
      }
  }
#endif

  // ------------ Run Bayesian Optimization
  printf("---- Define Bayesian Optimization Parameters ----\n");

  // Bayesian Optimization parameters
  bayesopt::Parameters bo_parameters = initialize_parameters_to_default();

  bo_parameters.n_iterations = config.wilson_BO_n_iter;
  bo_parameters.n_inner_iterations = config.wilson_BO_n_inner_iter;
  bo_parameters.n_init_samples = config.wilson_BO_init_samples;
  bo_parameters.n_iter_relearn = config.wilson_BO_n_inner_iter;
  bo_parameters.init_method = config.wilson_BO_init_method;
  bo_parameters.verbose_level = config.wilson_BO_verbose_level;
  bo_parameters.log_filename = config.wilson_BO_log_file;
  bo_parameters.surr_name = config.wilson_BO_surrogate;
  bo_parameters.sc_type = static_cast<score_type>(config.wilson_BO_sc_type);
  bo_parameters.l_type = static_cast<learning_type>(config.wilson_BO_l_type);
  bo_parameters.l_all = config.wilson_BO_l_all;
  bo_parameters.epsilon = config.wilson_BO_epsilon;
  bo_parameters.force_jump = config.wilson_BO_force_jump;
  bo_parameters.crit_name = config.wilson_BO_crit_name;

  // Call Bayesian Optimization
  // wilson_objective(2, NULL, NULL, NULL);
  const int num_dimensions = 2;
  double lower_bounds[num_dimensions] = { 0.0, 0.0 };
  double upper_bounds[num_dimensions] = { 1.0, 1.0 };

  double minimizer[num_dimensions] = { config.coupling_strength, config.delay };
  double minimizer_value[128];

  printf("---- Run Bayesian Optimization ----\n");
  int wilson_BO_output = bayes_optimization(num_dimensions,
                                            &wilson_objective,
                                            this, // can be used for pass class pointer if it will be a class
                                            lower_bounds,
                                            upper_bounds,
                                            minimizer,
                                            minimizer_value,
                                            bo_parameters.generate_bopt_params());

  // Note that the output of Bayesian Optimization will just be an error message, which we can output
  printf("---- Bayesian Optimization output ----\n");
  if (wilson_BO_output == 0) {
    printf("Bayesian Optimization was successful!\n");
  }
  else {
    printf("Bayesian Optimization was unsuccessful!. Output is %d\n", wilson_BO_output);
  }

  // Note that the output minimizer is stored in the minimizer array
  printf("---- Bayesian Optimization minimizer ----\n");
  for (int i = 0; i < num_dimensions; i++) {
    printf("Minimizer value for dimension %d is %f\n", i, minimizer[i]);
  }

  return wilson_electrical_activity;
}

/**
 * Define the objective function for the Wilson model
 * @param input_dim number of parameters
 * @param initial_query initial parameter values
 * @param gradient gradient of the objective function
 * @param func_data additional data for the objective function (which I think means data for the wilson model)
 * @return value of the objective function
 */
double Wilson::wilson_objective(unsigned int input_dim,
                                const double *initial_query,
                                double* gradient,
                                void *func_data)
{
    auto& instance = *(static_cast<Wilson*>(func_data));
    // ------------- Check the shape and type of global input variable values
    printf("---- Check global input variable values ----\n");
    //check_type((boost::any)::wilson_number_of_oscillators, "int * __ptr64", "wilson_number_of_oscillators");

    // ------------- Declare input variables - arrays
    printf("---- Declare helper variables ----\n");
    long temp_long; //
    double node_input;
    double delay_difference;
    int index_lower;
    int index_upper;
    double input_lower;
    double input_upper;
    double input_final;
    auto differential_E = std::vector<double>(instance.config.wilson_number_of_oscillators);
    auto differential_I = std::vector<double>(instance.config.wilson_number_of_oscillators);
    auto differential_E2 = std::vector<double>(instance.config.wilson_number_of_oscillators);
    auto differential_I2 = std::vector<double>(instance.config.wilson_number_of_oscillators);
    auto activity_E = std::vector<double>(instance.config.wilson_number_of_oscillators);
    auto activity_I = std::vector<double>(instance.config.wilson_number_of_oscillators);
    auto noises_array = std::vector<double>(instance.config.wilson_number_of_oscillators);
    instance.wilson_delay_mat =
      std::vector<std::vector<double>>(instance.config.wilson_number_of_oscillators,
                                       std::vector<double>(instance.config.wilson_number_of_oscillators));
    instance.wilson_coupling_mat =
      std::vector<std::vector<double>>(instance.config.wilson_number_of_oscillators,
                                       std::vector<double>(instance.config.wilson_number_of_oscillators));
    // ------------ Random generation
    std::default_random_engine generator(1);

    // ------------ TEMPORAL INTEGRATION
    printf("---- Temporal integration ----\n");
    for (int step = 1; step <= instance.config.wilson_number_of_integration_steps; step++)
    {
      if (step % 10000 == 0)
        printf("-- Temporal integration step %d --\n", step);
      // printf("-- Heun's Method - Step 1 --\n");
      // ------------ Heun's Method - Step 1
      for (int node = 0; node < instance.config.wilson_number_of_oscillators; node++)
      {
        // printf("-- Heun's 1: Node %d --\n", node);
        // ------------ Initializations
        // Initialize input to node as 0
        node_input = 0;

        // Initialize noise
        if (instance.config.wilson_noise_type == 0)
        {
          noises_array[node] = 0;
        }
        else if(instance.config.wilson_noise_type == 1)
        {
          noises_array[node] = instance.config.wilson_noise_amplitude * (2 * instance.rand_std_uniform(generator) - 1);
        }
        else if(instance.config.wilson_noise_type == 2)
        {
          noises_array[node] = instance.config.wilson_noise_amplitude * instance.rand_std_normal(generator);
        }

        // printf("-- Heun's 1: Node %d - Noise: %f --\n", node, noises_array[node]);

        // ------------ Calculate input to node
        // Consider all other nodes, but only if the lower delay index is lower than the time point
        for (int other_node = 0; other_node < instance.config.wilson_number_of_oscillators; other_node++)
        {
          // printf("-- Heun's 1: Node %d - Other node %d --\n", node, other_node);
          if (step > instance.config.wilson_lower_idxs_mat[node][other_node])
          {
            // Retrieve the difference between the 'true' delay and the one corresponding to the upper index
            delay_difference = instance.wilson_delay_mat[node][other_node];
            delay_difference -= (double)instance.config.wilson_upper_idxs_mat[node][other_node] *
                                        instance.config.wilson_integration_step_size;

            // Retrieve the time point indices corresponding with the lower and upper delay indices
            index_lower = step - 1 - instance.config.wilson_lower_idxs_mat[node][other_node];
            index_upper = step - 1 - instance.config.wilson_upper_idxs_mat[node][other_node];

            // Retrieve the activities corresponding to the lower and upper delay indices
            input_lower = instance.wilson_output_e[other_node][index_lower];
            input_upper = instance.wilson_output_e[other_node][index_upper];

            // From the previously retrieved values, estimate the input to oscillator k from oscillator j
            input_final = input_upper;
            input_final += (input_lower - input_upper) / instance.config.wilson_integration_step_size * delay_difference;
            // From this estimation, determine the quantile, final input
            input_final *= instance.wilson_coupling_mat[node][other_node];
            // Add this to the total input to oscillator k
            node_input += input_final;
          }
        }

        // printf("-- Heun's 1: Node %d - Differential Equations --\n", node);
        // ------------ Calculate Equations
        // Excitatory population (without noise and time) differentials
        differential_E[node] = node_input -
                               instance.config.wilson_c_ei * instance.config.wilson_i_values[node] -
                               instance.config.wilson_external_e;
        differential_E[node] = - instance.config.wilson_e_values[node] +
                                 (1 - instance.config.wilson_r_e * instance.config.wilson_e_values[node]) *
                                 wilson_response_function(differential_E[node], instance.config.wilson_alpha_e, instance.config.wilson_theta_e);

        // Inhibitory population (without noise and time) differentials
        differential_I[node] = instance.config.wilson_c_ie * instance.config.wilson_e_values[node];
        differential_I[node] = - instance.config.wilson_i_values[node] +
                                 (1 - instance.config.wilson_r_i * instance.config.wilson_i_values[node]) *
                                 wilson_response_function(differential_I[node], instance.config.wilson_alpha_i, instance.config.wilson_theta_i);

        // First estimate of the new activity values
        activity_E[node] = instance.config.wilson_e_values[node] +
                           (instance.config.wilson_integration_step_size * differential_E[node] +
                            sqrt(instance.config.wilson_integration_step_size) * noises_array[node]) / instance.config.wilson_tau_e;
        activity_I[node] = instance.config.wilson_i_values[node] +
                           (instance.config.wilson_integration_step_size * differential_I[node] +
                            sqrt(instance.config.wilson_integration_step_size) * noises_array[node]) / instance.config.wilson_tau_i;

        // printf("-- Heun's 1: Node %d - Update ::wilson_output_e value --\n", node);
        instance.wilson_output_e[node][step] = activity_E[node];
      }

      // printf("-- Heun's Method - Step 2 --\n");
      // ------------ Heun's Method - Step 2
      for(int node = 0; node < instance.config.wilson_number_of_oscillators; node++)
      {
        // printf("-- Heun's 2: Node %d --\n", node);
        // Initialize input to node as 0
        node_input = 0;

        // ------------ Calculate input to node
        // Consider all other nodes, but only if the lower delay index is lower than the time point
        for (int other_node = 0; other_node < instance.config.wilson_number_of_oscillators; other_node++)
        {
          // printf("-- Heun's 2: Node %d - Other node %d --\n", node, other_node);
          if (step > instance.config.wilson_lower_idxs_mat[node][other_node])
          {
            // printf("Step > lowerIdx");
            // Retrieve the difference between the 'true' delay and the one corresponding to the upper index
            delay_difference = instance.wilson_delay_mat[node][other_node];
            delay_difference -= (double)instance.config.wilson_upper_idxs_mat[node][other_node] *
                                        instance.config.wilson_integration_step_size;

            // Retrieve the time point indices corresponding with the lower and upper delay indices
            index_lower = step - instance.config.wilson_lower_idxs_mat[node][other_node];
            index_upper = step - instance.config.wilson_upper_idxs_mat[node][other_node];

            // Retrieve the activities corresponding to the lower and upper delay indices
            input_lower = instance.wilson_output_e[other_node][index_lower];
            input_upper = instance.wilson_output_e[other_node][index_upper];

            // From the previously retrieved values, estimate the input to oscillator k from oscillator j
            input_final = input_upper;
            input_final += (input_lower - input_upper) / instance.config.wilson_integration_step_size * delay_difference;
            // From this estimation, determine the quantile, final input
            input_final *= instance.wilson_coupling_mat[node][other_node];
            // Add this to the total input to oscillator k
            node_input += input_final;
          }
        }

        // printf("-- Heun's 2: Node %d - Differential Equations --\n", node);
        // ------------ Calculate Equations
        // Excitatory population (without noise and time) differentials
        differential_E2[node] = node_input - instance.config.wilson_c_ei * activity_I[node] + instance.config.wilson_external_e;
        differential_E2[node] = - activity_E[node] +
                                  (1 - instance.config.wilson_r_e * activity_E[node]) *
                                  wilson_response_function(differential_E2[node], instance.config.wilson_alpha_e, instance.config.wilson_theta_e);

        // Inhibitory population (without noise and time) differentials
        differential_I2[node] = instance.config.wilson_c_ie * activity_E[node];
        differential_I2[node] = - activity_I[node] +
                                  (1 - instance.config.wilson_r_i * activity_I[node]) *
                                  wilson_response_function(differential_I2[node], instance.config.wilson_alpha_i, instance.config.wilson_theta_i);

        // Second estimate of the new activity values
        instance.config.wilson_e_values[node] += (instance.config.wilson_integration_step_size /
                                                    2 * (differential_E[node] + differential_E2[node]) +
                                                  sqrt(instance.config.wilson_integration_step_size) * noises_array[node]) / instance.config.wilson_tau_e;
        instance.config.wilson_i_values[node] += (instance.config.wilson_integration_step_size /
                                                    2 * (differential_I[node] + differential_I2[node]) +
                                                  sqrt(instance.config.wilson_integration_step_size) * noises_array[node]) / instance.config.wilson_tau_i;

        // printf("-- Heun's 2: Node %d - Calculate ::wilson_output_e values --\n", node);
        instance.wilson_output_e[node][step] = instance.config.wilson_e_values[node];
      }
    }

    // ------------- Convert output variables to Python types
    printf("---- Converting output variables to Python types ----\n");
    instance.wilson_electrical_activity = instance.wilson_output_e;
#if 0 // TODO: Not needed anymore
    for (int i = 0; i < instance.wilson_number_of_oscillators; i++)
    {
      for (int step = 0; step <= instance.wilson_number_of_integration_steps; step++)
      {
        PyObject * temp_variable = PyFloat_FromDouble(::wilson_output_e[i * (*::wilson_number_of_integration_steps + 1) + step]);
        PyArray_SETITEM(::wilson_electrical_activity, PyArray_GETPTR2(::wilson_electrical_activity, i, step), temp_variable);
        // Decrease reference for next
        Py_DECREF(temp_variable);
      }
    }
#endif
    // ------------- Check that the output has the correct shape and type
    // printf("---- Check output variables ----\n");
    // check_type((boost::any)::wilson_electrical_activity, "struct __object * __ptr64", "wilson_electrical_activity");

    // ------------- Got electrical activity
    printf("---- Shape of electrical activity: %d x %d----\n",
           (int)instance.wilson_electrical_activity.size(),
           (int)instance.wilson_electrical_activity[0].size());

    // ------------- Convert the signal to BOLD
    printf("---- Converting electrical activity to BOLD ----\n");
    auto bold_signal = instance.electrical_to_bold(instance.wilson_electrical_activity,
                                                   instance.config.wilson_number_of_oscillators,
                                                   instance.config.wilson_number_of_integration_steps,
                                                   instance.config.wilson_integration_step_size);

    // Saving it just for a sanity check
    printf("----------- Saving unpacked BOLD signal -----------\n");
    saveBoldSignalToFile(bold_signal, "temp_arrays/unpacked_bold.csv");

    // TODO: It had better do that outside of this function by principe SOLID
    printf("----------- Filtering the BOLD signal -----------\n");
    std::vector<std::vector<double>> bold_filtered = process_BOLD(bold_signal,
                                                                  bold_signal.size(),
                                                                  bold_signal[0].size(),
                                                                  instance.config.wilson_order,
                                                                  instance.config.wilson_cutoffLow,
                                                                  instance.config.wilson_cutoffHigh,
                                                                  instance.config.wilson_sampling_rate);


    // Saving it just for a sanity check
    printf("----------- Saving filtered BOLD signal -----------\n");
    saveBoldSignalToFile(bold_filtered, "temp_arrays/filtered_bold.csv");

    // Printing shape of bold signal
    printf("---- Shape of BOLD signal: %zd x %zd----\n", bold_signal.size(), bold_signal[0].size());
    return 1;
}

//std::vector<double> to_vector_double(double *seq, int n)
//{
//  return std::vector<double>{seq, seq + n};
//}
//
//std::vector<std::vector<double>> to_vector_vector_double(double *seq, int n, int m)
//{
//  std::vector<std::vector<double>> data;
//  for (int i = 0; i < n; ++i) {
//    data[i] = std::vector<double>(seq + i * m, seq + i * m + n);
//  }
//  return data;
//}
//
//std::vector<int> to_vector_int(int *seq, int n)
//{
//  return std::vector<int>{seq, seq + n};
//}
//
//std::vector<std::vector<int>> to_vector_vector_int(int *seq, int n, int m)
//{
//  std::vector<std::vector<int>> data;
//  for (int i = 0; i < n; ++i) {
//    data[i] = std::vector<int>(seq + i * m, seq + i * m + n);
//  }
//  return data;
//}