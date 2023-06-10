// -------------------- Header file with some helper for the main simulation

#include <vector>
#include <stdexcept>
#include <boost/any.hpp>

// Function to check the data type of a variable
void check_type(boost::any input, const std::string& expected_type, const std::string& input_name) {

    // Get the type of the variable
    std::string true_type = input.type().name();
    printf("The type of %s is %s\n", input_name.c_str(), true_type.c_str());

    // Check if the type is the expected one
    if (true_type != expected_type) {
        throw std::invalid_argument("The type of " + input_name + " is " + true_type + " but it should be " + expected_type);
    }
    else {
        printf("The type of %s is correct\n", input_name.c_str());
    }
}

// Function to check the shape of a variable
template <typename T>
void check_shape(T const& input, std::vector<int>& expected_shape, std::string& input_name) {

    // Get the shape of the variable
    std::vector<int> true_shape = input.shape();

    // Check if the shape is the expected one
    if (true_shape != expected_shape) {
        throw std::invalid_argument("The shape of " + input_name + " is " + std::to_string(true_shape[0]) + "x" + std::to_string(true_shape[1]) + " but it should be " + std::to_string(expected_shape[0]) + "x" + std::to_string(expected_shape[1]));
    }
}

