#include "kbm_wrapper.h"

// Constructor for HodgkinHuxley
kbmWrapper::kbmWrapper(HodgkinHuxley* HodgkinHuxley_ptr, bool enable_HH_KBM_singular_output, float chosen_kbm_input_fraction) {
    HodgkinHuxleyInstance = HodgkinHuxley_ptr;
    HH_KBM_singular_output = enable_HH_KBM_singular_output; // default false
    kbm_input_fraction = chosen_kbm_input_fraction; // default 0
}

// Constructor for LorenzOscillator
kbmWrapper::kbmWrapper(LorenzOscillator* LorenzOscillator_ptr, float chosen_kbm_input_fraction) {
    LorenzOscillatorInstance = LorenzOscillator_ptr; // default false
    kbm_input_fraction = chosen_kbm_input_fraction; // default 0
}

Eigen::MatrixXd kbmWrapper::generateKBMfull(const Eigen::MatrixXd true_matrix, Eigen::MatrixXd forc_matrix) {

    // HodgkinHuxley
    if (HodgkinHuxleyInstance) {
        cout << "Generating HH KBM Full" << endl;

        // Shift forcing matrix to next-step to update variables.
        Eigen::VectorXd row = forc_matrix.row(0);
        std::vector<double> row_vector(row.data(), row.data() + row.size());
        row_vector.erase(row_vector.begin());
        HodgkinHuxleyInstance->stim = row_vector;
        

        for (int i = 0; i < true_matrix.cols()-1; ++i) {

            // Overwrite state variable with incoming signal 
            HodgkinHuxleyInstance->X[0] = true_matrix(0, i);

            if (true_matrix.rows() != 1) {
                HodgkinHuxleyInstance->X[1] = true_matrix(1, i);
                HodgkinHuxleyInstance->X[2] = true_matrix(2, i);
                HodgkinHuxleyInstance->X[3] = true_matrix(3, i);
            }

			// Runge-Kutta step forward
            HodgkinHuxleyInstance->rkStepper(i);

        }
        HodgkinHuxleyInstance->unpackTimeSeries();

        int n_cols = HodgkinHuxleyInstance->V_mem.cols();
        
        // Singular or All
        if (HH_KBM_singular_output) {
            //cout << "singsing enabled" << endl;
            Eigen::MatrixXd combined(1, n_cols);
            combined.row(0) = HodgkinHuxleyInstance->V_mem;
            return combined;
        }
        else
        {
            Eigen::MatrixXd combined(4, n_cols);
            combined.row(0) = HodgkinHuxleyInstance->V_mem;
            combined.row(1) = HodgkinHuxleyInstance->m_gate;
            combined.row(2) = HodgkinHuxleyInstance->h_gate;
            combined.row(3) = HodgkinHuxleyInstance->n_gate;
            return combined;
        }

    }

	// Lorenz Oscillator
    else if (LorenzOscillatorInstance) {
        cout << "Generating LZ KBM Full" << endl;

        for (int i = 0; i < true_matrix.cols(); ++i) {

            // Overwrite state variable with incoming signal 
            LorenzOscillatorInstance->X[0] = true_matrix(0, i);
            LorenzOscillatorInstance->X[1] = true_matrix(1, i);
            LorenzOscillatorInstance->X[2] = true_matrix(2, i);
            LorenzOscillatorInstance->rkStepper(i);

        }
        LorenzOscillatorInstance->unpackTimeSeries();

        int n_cols = LorenzOscillatorInstance->x_vals.cols();
        Eigen::MatrixXd combined(3, n_cols);
        combined.row(0) = LorenzOscillatorInstance->x_vals;
        combined.row(1) = LorenzOscillatorInstance->y_vals;
        combined.row(2) = LorenzOscillatorInstance->z_vals;

        return combined;
    }

    // Unhandled
    else
    {
        cerr << "Error: Unhandled DataGenerator type in KBM." << endl;
    }

}

Eigen::MatrixXd kbmWrapper::generateKBMlive(const Eigen::MatrixXd prediction, int index, int skip_length, int train_length) {
    
    // HodgkinHuxley
    if (HodgkinHuxleyInstance) {
        
        // Wipe forward-computing to be safe. Also helps with ease of data extraction for GSI.
        if (index == 0) {
            HodgkinHuxleyInstance->time_series.clear();
            HodgkinHuxleyInstance->time_series.resize(300000, vector<double>(6)); // overkill
            //cout << "Cleared " << endl;
        }

        // Always update voltage, update other state variables if predicted.
        HodgkinHuxleyInstance->X[0] = prediction(0, 0);
        if (prediction.rows() != 1) { 
            HodgkinHuxleyInstance->X[1] = prediction(1, 0);
            HodgkinHuxleyInstance->X[2] = prediction(2, 0);
            HodgkinHuxleyInstance->X[3] = prediction(3, 0);
        }

		// Runge-Kutta step forward
        HodgkinHuxleyInstance->rkStepper(skip_length + train_length + index - 1); 

        // Singular or All
        if (HH_KBM_singular_output) {
            //cout << "singsing" << endl;
            Eigen::MatrixXd combined(1, 1);
            combined(0, 0) = HodgkinHuxleyInstance->X[0];
            return combined;
        }
        else
        {
            Eigen::MatrixXd combined(4, 1);
            combined(0, 0) = HodgkinHuxleyInstance->X[0];
            combined(1, 0) = HodgkinHuxleyInstance->X[1];
            combined(2, 0) = HodgkinHuxleyInstance->X[2];
            combined(3, 0) = HodgkinHuxleyInstance->X[3];
            return combined;
        }

    }

    // Lorenz Oscillator
    else if (LorenzOscillatorInstance) {

        LorenzOscillatorInstance->X[0] = prediction(0, 0);
        LorenzOscillatorInstance->X[1] = prediction(1, 0);
        LorenzOscillatorInstance->X[2] = prediction(2, 0);
        LorenzOscillatorInstance->rkStepper(index); //check with parity.

        Eigen::MatrixXd combined(3, 1);
        combined(0, 0) = LorenzOscillatorInstance->X[0];
        combined(1, 0) = LorenzOscillatorInstance->X[1];
        combined(2, 0) = LorenzOscillatorInstance->X[2];

        return combined;
    }

    // Unhandled
    else
    {
        cerr << "Error: Unhandled DataGenerator type in KBM." << endl;
    }

}


