#pragma once

#include <random>
#include <vector>

//std::vector< std::vector<double> > to_vector_vector_double1(size_t len1_, size_t len2_, double* vec_);
//std::vector<double> to_vector_double1(size_t len_, double* vec_);
//std::vector< std::vector<int> > to_vector_vector_int1(size_t len1_, size_t len2_, int* vec_);
//std::vector<int> to_vector_int1(size_t len_, int* vec_);

//std::vector<double> to_vector_double1(double* vec_, size_t len_);

class INotifications {
public:
  INotifications() = default;

  // Required. Otherwise SWIG will issue warning.
  virtual ~INotifications() = default;

  // TODO: Not used for now just for example
  virtual void someNotification() = 0;
};

class WillsonConfig
{
  public:
    WillsonConfig() = default;

    enum class eScore {
      SC_MTL,
      SC_ML,
      SC_MAP,
      SC_LOOCV,
      SC_ERROR = -1
    };

    enum class eLearning {
      L_FIXED,
      L_EMPIRICAL,
      L_DISCRETE,
      L_MCMC,
      L_ERROR = -1
    };

//    void set_structural_connectivity_mat(size_t len1_, size_t len2_, double* vec_) {
//      structural_connectivity_mat = to_vector_vector_double1(len1_, len2_, vec_);
//    }
//
//    void set_wilson_lower_idxs_mat(size_t len1_, size_t len2_, int* vec_) {
//      wilson_lower_idxs_mat = to_vector_vector_int1(len1_, len2_, vec_);
//    }
//
//    void set_wilson_upper_idxs_mat(size_t len1_, size_t len2_, int* vec_) {
//      wilson_upper_idxs_mat = to_vector_vector_int1(len1_, len2_, vec_);
//    }

    double coupling_strength{}; ///< coupling strength
    double delay{}; ///< delay
    std::vector<std::vector<double>> structural_connectivity_mat{}; ///< structural connectivity
    int wilson_number_of_oscillators{};
    double wilson_c_ee{};
    double wilson_c_ei{};
    double wilson_c_ie{};
    double wilson_c_ii{};
    double wilson_tau_e{};
    double wilson_tau_i{};
    double wilson_r_e{};
    double wilson_r_i{};
    double wilson_alpha_e{};
    double wilson_alpha_i{};
    double wilson_theta_e{};
    double wilson_theta_i{};
    double wilson_external_e{};
    double wilson_external_i{};
    int wilson_number_of_integration_steps{};
    // TODO: wilson_integration_step_size may be should be int
    double wilson_integration_step_size{};
    std::vector<std::vector<double>> wilson_lower_idxs_mat{}; ///< lower_idxs matrix
    std::vector<std::vector<double>> wilson_upper_idxs_mat{}; ///< upper_idxs matrix
    std::vector<double> wilson_e_values{}; ///< initial conditions - EXCITATORY
    std::vector<double> wilson_i_values{}; ///< initial conditions - INHIBITORY
    // TODO: can be enum
    int wilson_noise_type{}; ///< noise type
    double wilson_noise_amplitude{}; ///< noise amplitude
    int wilson_order{}; ///< filter order
    double wilson_cutoffLow{}; ///< cutoff low
    double wilson_cutoffHigh{}; ///< cutoff high
    double wilson_sampling_rate{}; ///< sampling rate
    std::vector<std::vector<std::vector<double>>> emp_BOLD_signals{}; ///< BOLD signals
    int num_BOLD_subjects{}; ///< BOLD subjects
    int num_BOLD_regions{}; ///< BOLD regions
    int num_BOLD_timepoints{}; ///< BOLD timepoints
    int wilson_BO_n_iter{}; ///< BO iterations
    int wilson_BO_n_inner_iter{}; ///< BO inner iterations
    int wilson_BO_iter_relearn{}; ///< BO iterations before relearning
    int wilson_BO_init_samples{}; ///< BO initial samples
    int wilson_BO_init_method{}; ///< BO initial method
    int wilson_BO_verbose_level{}; ///< BO verbose level
    std::string wilson_BO_log_file{}; ///< BO log file
    std::string wilson_BO_surrogate{}; ///< BO surrogate
    eScore wilson_BO_sc_type{}; ///< BO score type
    eLearning wilson_BO_l_type{}; ///< BO learning type
    bool wilson_BO_l_all{}; ///< BO learn all
    double wilson_BO_epsilon{}; ///< BO epsilon
    int wilson_BO_force_jump{}; ///< BO force jump
    std::string wilson_BO_crit_name{}; ///< BO criterion name

    bool valid() const {
      if(wilson_e_values.size() != wilson_number_of_oscillators) return false;
      if(wilson_i_values.size() != wilson_number_of_oscillators) return false;
      if(wilson_lower_idxs_mat.size() != wilson_number_of_oscillators) return false;
      if(wilson_lower_idxs_mat[0].size() != wilson_number_of_oscillators) return false;
      if(wilson_upper_idxs_mat.size() != wilson_number_of_oscillators) return false;
      if(wilson_upper_idxs_mat[0].size() != wilson_number_of_oscillators) return false;

      if(emp_BOLD_signals.size() != num_BOLD_subjects) return false;
      if(emp_BOLD_signals[0].size() != num_BOLD_regions) return false;
      if(emp_BOLD_signals[0][0].size() != num_BOLD_timepoints) return false;

      if(structural_connectivity_mat.size() != wilson_number_of_oscillators) return false;
      if(structural_connectivity_mat[0].size() != wilson_number_of_oscillators) return false;
      return true;
    }
};

class Wilson {
public:
  explicit Wilson(WillsonConfig config);

  /** Run processing data.
   *
   * @return electrical activity of each node wrapped into numpy array
   */
  std::vector<std::vector<double>> process();

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
  std::vector<std::vector<double>> electrical_to_bold(std::vector<std::vector<double>>& wilson_electrical_activity,
                                                      int wilson_number_of_oscillators,
                                                      int wilson_number_of_integration_steps,
                                                      float wilson_integration_step_size);

  /**
   * Define the objective function for the Wilson model
   * @param input_dim number of parameters
   * @param initial_query initial parameter values
   * @param gradient gradient of the objective function
   * @param func_data additional data for the objective function (which I think means data for the wilson model)
   * @return value of the objective function
   */
  static double wilson_objective(unsigned int input_dim,
                                 const double *initial_query = nullptr,
                                 double* gradient = nullptr,
                                 void *func_data = nullptr);

private:
  WillsonConfig config;

  std::normal_distribution<double> rand_std_normal{0, 1};
  std::uniform_real_distribution<double> rand_std_uniform{0, 1};
  std::vector<std::vector<double>> wilson_electrical_activity{};
  std::vector<std::vector<double>> wilson_output_e{}; // TODO: Can be the same as wilson_electrical_activity
  //std::vector<std::vector<int64_t>> wilson_lower_idxs_mat{};
  //std::vector<std::vector<int64_t>> wilson_upper_idxs_mat{};

  std::vector<std::vector<double>> wilson_delay_mat;
  std::vector<std::vector<double>> wilson_coupling_mat;
};
