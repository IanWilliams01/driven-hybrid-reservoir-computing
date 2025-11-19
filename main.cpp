#include "config.h"
#include "utils.h"
#include "data_HodgkinHuxley.h"
#include "data_LorenzOscillator.h"
#include "machine_learning.cpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>

using namespace std;

// Data Generators-----------------------------------------------


void datagen_HH() {

    HodgkinHuxley hh_model(HH_diagnostic);
    hh_model.runModel();
    hh_model.exportResults();

}

void datagen_LZ() {

    LorenzOscillator lorenz_model(LZ_diagnostic);
    lorenz_model.runModel();
    lorenz_model.exportResults();

}


// Model-Error Correction (MEC) ---------------------------------

void readings_KBM_MEC_gNaT() {

    // Define Storage --------------------------------------------

    string results_folder = "output/102_KBM_MEC_gNaT/";
    string log_filename = results_folder + "00_trial_parameters.csv";
    string log_header =
        "Trial Number,"
        "Spectral Radius,"
        "Degree,"
        "Sigma,"
        "Beta,"
        "Approx Size,"
        "Train Length,"
        "Prediction Length,"
        "Skip Length,"
        "Scaling,"
        "Input_Hybrid,"
        "Output_Hybrid,"
        "Error,"
        "MAE,"
        "RMSE";
    string log_optional_comments = "";

    // Define Trial Space ----------------------------------------

    HH_ParamStruct true_HH = HH_trial;
    HH_ParamStruct unedited_kbm = HH_trial;
    vector<double> error_values = { 0.01, 0.1, 1 };
    vector<int> skip_length_values = { 0, 25000, 50000, 75000, 100000, 125000, 150000, 175000 };
    int repeat_readings = 3;


    // Execute ---------------------------------------------------

    HodgkinHuxley hh_model(true_HH);
    hh_model.runModel();

    int trial_number = 1;
    for (double error : error_values) {
        for (double skip_length : skip_length_values) {
            for (int i = 0; i < repeat_readings; ++i) {

                cout << "\nRun: " << trial_number << endl;
                cout << "------------------------------------------------" << endl;

                // Customise trial parameter set. (Explicitly defined for tinker-ability)
                RC_ParamStruct trial_RC_params = {
                    1,			    // spectral_radius (Stability)
                    8,			    // degree (Connectivity)
                    0.8,		    // sigma (Input responsiveness.)
                    0.001,		    // beta (Ridge regression regularisation)
                    1000,		    // approx_size (Approximate size of reservoir
                    50000,		    // train_length;			
                    25000,		    // predict_length;		
                    skip_length,	// skip_length
                    false,		    // scaling;
                };

                // Customise trial KBM params.
                HH_ParamStruct trial_kbm = unedited_kbm; // Reset to default values.
                trial_kbm.gNaT = trial_kbm.gNaT * (1 + error);
                //trial_kbm.amV1 = trial_kbm.amV1 * (1 + error);
                //trial_kbm.tm0 = trial_kbm.tm0 * (1 + error);

                // Setup KBM
                HodgkinHuxley HH_kbm(trial_kbm);
                kbmWrapper HH_kbmWrapper(&HH_kbm);

                // Run ML
                ReservoirComputing reservoir(trial_RC_params, hh_model.V_mem, hh_model.I_stim, &HH_kbmWrapper);
                reservoir.rng_seed += trial_number; // Change RNG for repeat readings.
                reservoir.trainReservoir();
                reservoir.generatePredictions();

                // Export Data
                string trial_filename = results_folder + "trial_" + to_string(trial_number) + ".csv";
                exportPredictionsToCSV(
                    reservoir.data_predict,
                    reservoir.predictions,
                    trial_filename
                );

                // Log Performance
                string trial_row =
                    to_string(trial_number) + ", " +
                    to_string(trial_RC_params.spectral_radius) + ", " +
                    to_string(trial_RC_params.degree) + ", " +
                    to_string(trial_RC_params.sigma) + ", " +
                    to_string(trial_RC_params.beta) + ", " +
                    to_string(trial_RC_params.approx_size) + ", " +
                    to_string(trial_RC_params.train_length) + ", " +
                    to_string(trial_RC_params.predict_length) + ", " +
                    to_string(trial_RC_params.skip_length) + ", " +
                    to_string(trial_RC_params.scaling) + ", " +
                    to_string(trial_RC_params.input_hybrid) + ", " +
                    to_string(trial_RC_params.output_hybrid) + ", " +
                    to_string(error) + ", " +
                    to_string(reservoir.predictions_mae) + ", " +
                    to_string(reservoir.predictions_rmse);

                logGenericTrial(
                    log_filename,
                    log_header,
                    trial_number,
                    trial_row
                );
                trial_number++;

            }
        }
    }

    // Log Comments.
    logGenericTrial(
        log_filename,
        log_header,
        -1,
        log_optional_comments
    );
}

void readings_KBM_MEC_amV1() {

    // Define Storage --------------------------------------------

    string results_folder = "output/chaos/27_KBM_MEC_amV1_ext/";
    string log_filename = results_folder + "00_trial_parameters.csv";
    string log_header =
        "Trial Number,"
        "Spectral Radius,"
        "Degree,"
        "Sigma,"
        "Beta,"
        "Approx Size,"
        "Train Length,"
        "Prediction Length,"
        "Skip Length,"
        "Scaling,"
        "Input_Hybrid,"
        "Output_Hybrid,"
        "Error,"
        "MAE,"
        "RMSE";
    string log_optional_comments = "";

    // Define Trial Space ----------------------------------------

    HH_ParamStruct true_HH = HH_trial;
    HH_ParamStruct unedited_kbm = HH_trial;
    vector<double> error_values = { 0 };
    vector<int> skip_length_values = { 0 };
    int repeat_readings = 1;


    // Execute ---------------------------------------------------

    HodgkinHuxley hh_model(true_HH);
    hh_model.runModel();

    int trial_number = 1;
    for (double error : error_values) {
        for (double skip_length : skip_length_values) {
            for (int i = 0; i < repeat_readings; ++i) {

                cout << "\nRun: " << trial_number << endl;
                cout << "------------------------------------------------" << endl;

                // Customise trial parameter set. (Explicitly defined for tinker-ability)
                RC_ParamStruct trial_RC_params = {
                    1,			    // spectral_radius (Stability)
                    8,			    // degree (Connectivity)
                    0.8,		    // sigma (Input responsiveness.)
                    0.001,		    // beta (Ridge regression regularisation)
                    1000,		    // approx_size (Approximate size of reservoir
                    50000,		    // train_length;			
                    200000,		    // predict_length;		
                    skip_length,	// skip_length
                    false,		    // scaling;
                };

                // Customise trial KBM params.
                HH_ParamStruct trial_kbm = unedited_kbm; // Reset to default values.
                //trial_kbm.gNaT = trial_kbm.gNaT * (1 + error);
                trial_kbm.amV1 = trial_kbm.amV1 * (1 + error);
                //trial_kbm.tm0 = trial_kbm.tm0 * (1 + error);

                // Setup KBM
                HodgkinHuxley HH_kbm(trial_kbm);
                kbmWrapper HH_kbmWrapper(&HH_kbm);

                // Run ML
                ReservoirComputing reservoir(trial_RC_params, hh_model.V_mem, hh_model.I_stim, &HH_kbmWrapper);
                reservoir.rng_seed += trial_number; // Change RNG for repeat readings.
                reservoir.trainReservoir();
                reservoir.generatePredictions();

                // Export Data
                string trial_filename = results_folder + "trial_" + to_string(trial_number) + ".csv";
                exportPredictionsToCSV(
                    reservoir.data_predict,
                    reservoir.predictions,
                    trial_filename
                );

                // Log Performance
                string trial_row =
                    to_string(trial_number) + ", " +
                    to_string(trial_RC_params.spectral_radius) + ", " +
                    to_string(trial_RC_params.degree) + ", " +
                    to_string(trial_RC_params.sigma) + ", " +
                    to_string(trial_RC_params.beta) + ", " +
                    to_string(trial_RC_params.approx_size) + ", " +
                    to_string(trial_RC_params.train_length) + ", " +
                    to_string(trial_RC_params.predict_length) + ", " +
                    to_string(trial_RC_params.skip_length) + ", " +
                    to_string(trial_RC_params.scaling) + ", " +
                    to_string(trial_RC_params.input_hybrid) + ", " +
                    to_string(trial_RC_params.output_hybrid) + ", " +
                    to_string(error) + ", " +
                    to_string(reservoir.predictions_mae) + ", " +
                    to_string(reservoir.predictions_rmse);

                logGenericTrial(
                    log_filename,
                    log_header,
                    trial_number,
                    trial_row
                );
                trial_number++;

            }
        }
    }

    // Log Comments.
    logGenericTrial(
        log_filename,
        log_header,
        -1,
        log_optional_comments
    );
}

void readings_KBM_MEC_tm0() {

    // Define Storage --------------------------------------------

    string results_folder = "output/53b_KBM_MEC_tm0/";
    string log_filename = results_folder + "00_trial_parameters.csv";
    string log_header =
        "Trial Number,"
        "Spectral Radius,"
        "Degree,"
        "Sigma,"
        "Beta,"
        "Approx Size,"
        "Train Length,"
        "Prediction Length,"
        "Skip Length,"
        "Scaling,"
        "Input_Hybrid,"
        "Output_Hybrid,"
        "Error,"
        "MAE,"
        "RMSE";
    string log_optional_comments = "";

    // Define Trial Space ----------------------------------------

    HH_ParamStruct true_HH = HH_trial;
    HH_ParamStruct unedited_kbm = HH_trial;
    vector<double> error_values = { 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 };
    vector<int> skip_length_values = { 0, 25000, 50000, 75000, 100000, 125000, 150000, 175000 };
    int repeat_readings = 1;


    // Execute ---------------------------------------------------

    HodgkinHuxley hh_model(true_HH);
    hh_model.runModel();

    int trial_number = 1;
    for (double error : error_values) {
        for (double skip_length : skip_length_values) {
            for (int i = 0; i < repeat_readings; ++i) {

                cout << "\nRun: " << trial_number << endl;
                cout << "------------------------------------------------" << endl;

                // Customise trial parameter set. (Explicitly defined for tinker-ability)
                RC_ParamStruct trial_RC_params = {
                    1,			    // spectral_radius (Stability)
                    8,			    // degree (Connectivity)
                    0.8,		    // sigma (Input responsiveness.)
                    0.001,		    // beta (Ridge regression regularisation)
                    1000,		    // approx_size (Approximate size of reservoir
                    50000,		    // train_length;			
                    25000,		    // predict_length;		
                    skip_length,	// skip_length
                    false,		    // scaling;
                };

                // Customise trial KBM params.
                HH_ParamStruct trial_kbm = unedited_kbm; // Reset to default values.
                //trial_kbm.gNaT = trial_kbm.gNaT * (1 + error);
                //trial_kbm.amV1 = trial_kbm.amV1 * (1 + error);
                trial_kbm.tm0 = trial_kbm.tm0 * (1 + error);

                // Setup KBM
                HodgkinHuxley HH_kbm(trial_kbm);
                kbmWrapper HH_kbmWrapper(&HH_kbm);

                // Run ML
                ReservoirComputing reservoir(trial_RC_params, hh_model.V_mem, hh_model.I_stim, &HH_kbmWrapper);
                reservoir.rng_seed += trial_number; // Change RNG for repeat readings.
                reservoir.trainReservoir();
                reservoir.generatePredictions();

                // Export Data
                string trial_filename = results_folder + "trial_" + to_string(trial_number) + ".csv";
                exportPredictionsToCSV(
                    reservoir.data_predict,
                    reservoir.predictions,
                    trial_filename
                );

                // Log Performance
                string trial_row =
                    to_string(trial_number) + ", " +
                    to_string(trial_RC_params.spectral_radius) + ", " +
                    to_string(trial_RC_params.degree) + ", " +
                    to_string(trial_RC_params.sigma) + ", " +
                    to_string(trial_RC_params.beta) + ", " +
                    to_string(trial_RC_params.approx_size) + ", " +
                    to_string(trial_RC_params.train_length) + ", " +
                    to_string(trial_RC_params.predict_length) + ", " +
                    to_string(trial_RC_params.skip_length) + ", " +
                    to_string(trial_RC_params.scaling) + ", " +
                    to_string(trial_RC_params.input_hybrid) + ", " +
                    to_string(trial_RC_params.output_hybrid) + ", " +
                    to_string(error) + ", " +
                    to_string(reservoir.predictions_mae) + ", " +
                    to_string(reservoir.predictions_rmse);

                logGenericTrial(
                    log_filename,
                    log_header,
                    trial_number,
                    trial_row
                );
                trial_number++;

            }
        }
    }

    // Log Comments.
    logGenericTrial(
        log_filename,
        log_header,
        -1,
        log_optional_comments
    );
}

void readings_KBM_MEC_3x() {

    // Define Storage --------------------------------------------

    string results_folder = "output/54_KBM_MEC_combined/";
    string log_filename = results_folder + "00_trial_parameters.csv";
    string log_header =
        "Trial Number,"
        "Spectral Radius,"
        "Degree,"
        "Sigma,"
        "Beta,"
        "Approx Size,"
        "Train Length,"
        "Prediction Length,"
        "Skip Length,"
        "Scaling,"
        "Input_Hybrid,"
        "Output_Hybrid,"
        "Error,"
        "MAE,"
        "RMSE";
    string log_optional_comments = "";

    // Define Trial Space ----------------------------------------

    HH_ParamStruct true_HH = HH_trial;
    HH_ParamStruct unedited_kbm = HH_trial;
    vector<double> error_values = { 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 };
    vector<int> skip_length_values = { 0, 25000, 50000, 75000, 100000, 125000, 150000, 175000 };
    int repeat_readings = 4;


    // Execute ---------------------------------------------------

    HodgkinHuxley hh_model(true_HH);
    hh_model.runModel();

    int trial_number = 1;
    for (double error : error_values) {
        for (double skip_length : skip_length_values) {
            for (int i = 0; i < repeat_readings; ++i) {

                cout << "\nRun: " << trial_number << endl;
                cout << "------------------------------------------------" << endl;

                // Customise trial parameter set. (Explicitly defined for tinker-ability)
                RC_ParamStruct trial_RC_params = {
                    1,			    // spectral_radius (Stability)
                    8,			    // degree (Connectivity)
                    0.8,		    // sigma (Input responsiveness.)
                    0.001,		    // beta (Ridge regression regularisation)
                    1000,		    // approx_size (Approximate size of reservoir
                    50000,		    // train_length;			
                    25000,		    // predict_length;		
                    skip_length,	// skip_length
                    false,		    // scaling;
                };

                // Customise trial KBM params.
                HH_ParamStruct trial_kbm = unedited_kbm; // Reset to default values.
                trial_kbm.gNaT = trial_kbm.gNaT * (1 + error);
                trial_kbm.amV1 = trial_kbm.amV1 * (1 + error);
                trial_kbm.tm0 = trial_kbm.tm0 * (1 + error);

                // Setup KBM
                HodgkinHuxley HH_kbm(trial_kbm);
                kbmWrapper HH_kbmWrapper(&HH_kbm);

                // Run ML
                ReservoirComputing reservoir(trial_RC_params, hh_model.V_mem, hh_model.I_stim, &HH_kbmWrapper);
                reservoir.rng_seed += trial_number; // Change RNG for repeat readings.
                reservoir.trainReservoir();
                reservoir.generatePredictions();

                // Export Data
                string trial_filename = results_folder + "trial_" + to_string(trial_number) + ".csv";
                exportPredictionsToCSV(
                    reservoir.data_predict,
                    reservoir.predictions,
                    trial_filename
                );

                // Log Performance
                string trial_row =
                    to_string(trial_number) + ", " +
                    to_string(trial_RC_params.spectral_radius) + ", " +
                    to_string(trial_RC_params.degree) + ", " +
                    to_string(trial_RC_params.sigma) + ", " +
                    to_string(trial_RC_params.beta) + ", " +
                    to_string(trial_RC_params.approx_size) + ", " +
                    to_string(trial_RC_params.train_length) + ", " +
                    to_string(trial_RC_params.predict_length) + ", " +
                    to_string(trial_RC_params.skip_length) + ", " +
                    to_string(trial_RC_params.scaling) + ", " +
                    to_string(trial_RC_params.input_hybrid) + ", " +
                    to_string(trial_RC_params.output_hybrid) + ", " +
                    to_string(error) + ", " +
                    to_string(reservoir.predictions_mae) + ", " +
                    to_string(reservoir.predictions_rmse);

                logGenericTrial(
                    log_filename,
                    log_header,
                    trial_number,
                    trial_row
                );
                trial_number++;

            }
        }
    }

    // Log Comments.
    logGenericTrial(
        log_filename,
        log_header,
        -1,
        log_optional_comments
    );
}


// Gate-State Inference (GSI) -----------------------------------

void kbm_internal_variables0() {

    // Generate Data
    HodgkinHuxley hh_model(HH_trial);
    hh_model.runModel();
    hh_model.export_filepath = "output/76_KBM_internal_variables_0_1%_err_gNaT/HH_true.csv";
    hh_model.exportResults();

    // Setup KBM
    HH_trial.gNaT = HH_trial.gNaT * (1 + 0.001);
    HodgkinHuxley HH_kbm(HH_trial);
    kbmWrapper HH_kbmWrapper(&HH_kbm);

    // Run ML
    ReservoirComputing reservoir(RC_hybrid_internal_variables, hh_model.V_mem, hh_model.I_stim, &HH_kbmWrapper);
    reservoir.trainReservoir();
    reservoir.generatePredictions();


    HH_kbm.export_filepath = "output/76_KBM_internal_variables_0_1%_err_gNaT/hybrid_pred.csv";
    HH_kbm.exportResults();

}


void readings_erroneous_datagen() {

    string filepath = "output/35_KBM_Alone_combined_error/";

    HodgkinHuxley hh_model(HH_trial);
    hh_model.export_filepath = filepath + "0.csv";
    hh_model.runModel();
    hh_model.exportResults();

    vector<double> error_values = { 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 };

    for (double error : error_values) {
    
        HH_ParamStruct trial_kbm = HH_trial; // Reset to default values.

        trial_kbm.gNaT = trial_kbm.gNaT * (1 + error);
        trial_kbm.amV1 = trial_kbm.amV1 * (1 + error);
        trial_kbm.tm0 = trial_kbm.tm0 * (1 + error);

        HodgkinHuxley HH_kbm(trial_kbm);

        string filename = to_string(error);
        HH_kbm.export_filepath = filepath + filename + ".csv";

        HH_kbm.runModel();
        HH_kbm.exportResults();
    
    }

}


// Demo Testbeds ------------------------------------------------


// HH RC Simple.
void testbed_alpha() {
    
    // Generate Data
    HodgkinHuxley hh_model(HH_trial);
    hh_model.runModel();

    // Run ML
    ReservoirComputing reservoir(RC_trial_HH_adjopt, hh_model.V_mem, hh_model.I_stim);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_alpha.csv");

}

// LZ RC Simple.
void testbed_bravo() {

    // Generate Data
    LorenzOscillator lorenz_model(LZ_diagnostic);
    lorenz_model.runModel();

    Eigen::MatrixXd combined(3, lorenz_model.x_vals.cols());
    combined.row(0) = lorenz_model.x_vals;
    combined.row(1) = lorenz_model.y_vals;
    combined.row(2) = lorenz_model.z_vals;

    // Setup Forc
    Eigen::MatrixXd zeroMatrix = Eigen::MatrixXd::Zero(0, 0);
    
    // Run ML
    ReservoirComputing reservoir(RC_trial_LZ_Test, combined, zeroMatrix);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_bravo.csv");

}

// HH RC Multi.
void testbed_charlie() {

    // Generate Data
    HodgkinHuxley hh_model(HH_diagnostic);
    hh_model.runModel();

    // Set each row of the combined matrix
    Eigen::MatrixXd combined(4, hh_model.V_mem.cols());
    combined.row(0) = hh_model.V_mem;
    combined.row(1) = hh_model.m_gate;
    combined.row(2) = hh_model.h_gate;
    combined.row(3) = hh_model.n_gate;

    // Run ML
    ReservoirComputing reservoir(RC_trial_HH_multi, combined, hh_model.I_stim);
    reservoir.trainReservoir();
    reservoir.generatePredictions();


    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_charlie.csv");

}

// HH RC Gaussian Error
void testbed_delta() {

    // Generate Data
    HodgkinHuxley hh_model(HH_diagnostic);
    hh_model.runModel();

    // Apply noise
    applyGaussianNoise(hh_model.V_mem, 0.1);

    // Run ML
    ReservoirComputing reservoir(RC_diagnostic, hh_model.V_mem, hh_model.I_stim);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_delta.csv");

}

// LZ KBM ON (Paired Below w/ Foxtrot)
void testbed_echo() {

    // Generate Data
    LorenzOscillator lorenz_model(lorenz_trial);
    lorenz_model.runModel();

    Eigen::MatrixXd combined(3, lorenz_model.x_vals.cols());
    combined.row(0) = lorenz_model.x_vals;
    combined.row(1) = lorenz_model.y_vals;
    combined.row(2) = lorenz_model.z_vals;

    // Setup Forc
    Eigen::MatrixXd zeroMatrix = Eigen::MatrixXd::Zero(0, 0);

    // Setup KBM
    LorenzOscillator lorenz_kbm(lorenz_trial_kbm);
    kbmWrapper lorenz_kbmWrapper(&lorenz_kbm);

    // Run ML
    ReservoirComputing reservoir(RC_trial_LZ_kbm_Test, combined, zeroMatrix, &lorenz_kbmWrapper);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_echo.csv");

    //lorenz_model.exportResults();
    //lorenz_kbm.exportResults();

}

// LZ KBM OFF (Paired Above w/ Echo)
void testbed_foxtrot() {

    // Generate Data
    LorenzOscillator lorenz_model(lorenz_trial);
    lorenz_model.runModel();

    Eigen::MatrixXd combined(3, lorenz_model.x_vals.cols());
    combined.row(0) = lorenz_model.x_vals;
    combined.row(1) = lorenz_model.y_vals;
    combined.row(2) = lorenz_model.z_vals;

    // Setup Forc
    Eigen::MatrixXd zeroMatrix = Eigen::MatrixXd::Zero(0, 0);

    // Run ML
    ReservoirComputing reservoir(RC_trial_LZ_kbm_Test, combined, zeroMatrix);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_foxtrot.csv");

}

// --------------------------------------------------
// HH KBM ON Multi-Multi (Paired Below w/ Hotel)
void testbed_golf() {

    // Generate Data
    HodgkinHuxley hh_model(HH_trial);
    hh_model.runModel();

    // Set each row of the combined matrix
    Eigen::MatrixXd combined(4, hh_model.V_mem.cols());
    combined.row(0) = hh_model.V_mem;
    combined.row(1) = hh_model.m_gate;
    combined.row(2) = hh_model.h_gate;
    combined.row(3) = hh_model.n_gate;

    // Setup KBM
    HodgkinHuxley HH_kbm(HH_trial_kbm);
    kbmWrapper HH_kbmWrapper(&HH_kbm);

    // Run ML
    ReservoirComputing reservoir(RC_trial_HH_kbm_test, combined, hh_model.I_stim, &HH_kbmWrapper);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_golf.csv");


}

// HH KBM OFF Multi-Multi (Paired Above w/ Golf)
void testbed_hotel() {

    // Generate Data
    HodgkinHuxley hh_model(HH_trial);
    hh_model.runModel();

    // Set each row of the combined matrix
    Eigen::MatrixXd combined(4, hh_model.V_mem.cols());
    combined.row(0) = hh_model.V_mem;
    combined.row(1) = hh_model.m_gate;
    combined.row(2) = hh_model.h_gate;
    combined.row(3) = hh_model.n_gate;

    // Run ML
    ReservoirComputing reservoir(RC_trial_HH_kbm_test, combined, hh_model.I_stim);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_hotel.csv");

}

// --------------------------------------------------
// HH KBM ON Sing-Multi (Paired Below w/ Juliet)
void testbed_indigo() {

    // Generate Data
    HodgkinHuxley hh_model(HH_trial);
    hh_model.runModel();

    // Setup KBM
    HodgkinHuxley HH_kbm(HH_trial_kbm);
    kbmWrapper HH_kbmWrapper(&HH_kbm);

    // Run ML
    ReservoirComputing reservoir(RC_trial_HH_kbm_test, hh_model.V_mem, hh_model.I_stim, &HH_kbmWrapper);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_indigo.csv");

}

// HH KBM OFF Sing-Multi (Paired Above w/ Indigo)
void testbed_juliet() {

    // Generate Data
    HodgkinHuxley hh_model(HH_trial);
    hh_model.runModel();

    // Run ML
    ReservoirComputing reservoir(RC_trial_HH_kbm_test, hh_model.V_mem, hh_model.I_stim);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_juliet.csv");

}

// --------------------------------------------------
// HH KBM ON Sing-Sing (Paired Below w/ Lima)
void testbed_kilo() {

    // Generate Data
    HodgkinHuxley hh_model(HH_trial);
    hh_model.runModel();

    // Setup KBM
    HodgkinHuxley HH_kbm(HH_trial_kbm);
    kbmWrapper HH_kbmWrapper(&HH_kbm, true);

    // Run ML 
    ReservoirComputing reservoir(RC_trial_HH_kbm_test, hh_model.V_mem, hh_model.I_stim, &HH_kbmWrapper);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_kilo.csv");

}

// HH KBM OFF Sing-Sing (Paired Above w/ kilo)
void testbed_lima() {

    // Generate Data
    HodgkinHuxley hh_model(HH_trial);
    hh_model.runModel();

    // Run ML
    ReservoirComputing reservoir(RC_trial_HH_adjopt, hh_model.V_mem, hh_model.I_stim);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_lima.csv");

}

// --------------------------------------------------
// HH KBM ON Multi-Sing (Paired Below w/ November)
void testbed_mike() {

    // Generate Data
    HodgkinHuxley hh_model(HH_trial);
    hh_model.runModel();

    // Set each row of the combined matrix
    Eigen::MatrixXd combined(4, hh_model.V_mem.cols());
    combined.row(0) = hh_model.V_mem;
    combined.row(1) = hh_model.m_gate;
    combined.row(2) = hh_model.h_gate;
    combined.row(3) = hh_model.n_gate;

    // Setup KBM
    HodgkinHuxley HH_kbm(HH_trial_kbm);
    kbmWrapper HH_kbmWrapper(&HH_kbm, true);

    // Run ML
    ReservoirComputing reservoir(RC_trial_HH_kbm_test, combined, hh_model.I_stim, &HH_kbmWrapper);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_mike.csv");


}

// HH KBM OFF Multi-Sing (Paired Above w/ Mike)
void testbed_november() {

    // Generate Data
    HodgkinHuxley hh_model(HH_trial);
    hh_model.runModel();

    // Set each row of the combined matrix
    Eigen::MatrixXd combined(4, hh_model.V_mem.cols());
    combined.row(0) = hh_model.V_mem;
    combined.row(1) = hh_model.m_gate;
    combined.row(2) = hh_model.h_gate;
    combined.row(3) = hh_model.n_gate;

    // Run ML
    ReservoirComputing reservoir(RC_trial_HH_kbm_test, combined, hh_model.I_stim);
    reservoir.trainReservoir();
    reservoir.generatePredictions();

    // Export Results
    exportPredictionsToCSV(reservoir.data_predict, reservoir.predictions, "output/testbed_november.csv");

}


// ----------------------------------------------------------------------------------------------------
// Main.


int main() {

    // -------------------------

    //datagen_HH();
    //datagen_LZ();

    // -------------------------

    readings_KBM_MEC_gNaT();
    //readings_KBM_MEC_amV1();
    //readings_KBM_MEC_tm0();
    //readings_KBM_MEC_3x();

    // -------------------------

    //kbm_internal_variables0();
    //readings_erroneous_datagen();

    // -------------------------

    //testbed_alpha();
    //testbed_bravo();
    //testbed_charlie();
    //testbed_delta();
    //testbed_echo();
    //testbed_foxtrot();
    //testbed_golf();
    //testbed_hotel();
    //testbed_indigo();
    //testbed_juliet();
    //testbed_kilo();
    //testbed_lima();
    //testbed_mike();
    //testbed_november();

    // -------------------------

    
}

