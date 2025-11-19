#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <Eigen/Dense>
#include <random>

#include "config_types.h"

using namespace std;

// ----------------------------------------------------------------------------------------------------


void progressBar(int current_step, int total_steps, int update_frequency = 25);


void testTrainSplit(
    const Eigen::MatrixXd& input_matrix,
    int train_length,
    int prediction_length,
    Eigen::MatrixXd& train_matrix,
    Eigen::MatrixXd& predict_matrix,
    int skip_n = 0);

// Function to write data to a CSV file
void writeVectorofVectorToCSV(string& filename, vector<vector<double>>& output_data, string& header);


// ----------------------------------------------------------------------------------------------------


void writeMatrixToCSV(const std::string& filepath, const Eigen::MatrixXd& matrix);

void exportPredictionsToCSV(
    const Eigen::MatrixXd& matrix1,
    const Eigen::MatrixXd& matrix2,
    const string& filename,
    const string& prefix1 = "True",
    const string& prefix2 = "Preds"
);


void logGridSearchTrialParams(
    const string& filename,
    int trial_number,
    const RC_ParamStruct& params,
    double trial_mae,
    double trial_rmse
);


void logGenericTrial(
    const string& filename,
    string header,
    int trial_number,
    string trial_row
);

// ----------------------------------------------------------------------------------------------------

void applyGaussianNoise(Eigen::MatrixXd& matrix, double std_noise, int rng_seed = 42);


class Scaler {
private:
    Eigen::VectorXd mean;
    Eigen::VectorXd std_dev;
public:
    Scaler() = default;
    void fit(const Eigen::MatrixXd& data);
    Eigen::MatrixXd scale(const Eigen::MatrixXd& data) const;
    Eigen::MatrixXd unscale(const Eigen::MatrixXd& scaledData) const;
};