#ifndef KBMWRAPPER_H
#define KBMWRAPPER_H

#include "config_types.h"
#include "utils.h"
#include "data_HodgkinHuxley.h"
#include "data_LorenzOscillator.h"

#include <Eigen/Dense> 
#include <optional>
#include <iostream>

class kbmWrapper {
private:

    // Define nullptrs, one of which will be set by constructor.
    HodgkinHuxley* HodgkinHuxleyInstance = nullptr;
    LorenzOscillator* LorenzOscillatorInstance = nullptr;

    bool HH_KBM_singular_output = false;
    

public:

    float kbm_input_fraction = 0;

    // Constructor for HodgkinHuxley
    kbmWrapper(HodgkinHuxley* HodgkinHuxley_ptr, bool enable_HH_KBM_singular_output = false, float chosen_kbm_input_fraction = 0);

    // Constructor for LorenzOscillator
    kbmWrapper(LorenzOscillator* LorenzOscillator_ptr, float chosen_kbm_input_fraction = 0);

    ~kbmWrapper() = default;

    // ----------------------------------------------------------------------------------------------------
    // Generators

    Eigen::MatrixXd generateKBMfull(const Eigen::MatrixXd true_matrix, Eigen::MatrixXd forc_matrix);

    Eigen::MatrixXd generateKBMlive(const Eigen::MatrixXd prediction, int index, int skip_length, int train_length);
};

#endif // KBMWRAPPER_H
