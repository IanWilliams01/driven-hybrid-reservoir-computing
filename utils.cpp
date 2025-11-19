#pragma once

#include "utils.h"

using namespace std;

// ----------------------------------------------------------------------------------------------------

void progressBar(int current_step, int total_steps, int update_frequency) {

    if (current_step % (total_steps / (100 / update_frequency)) == 0) {
        int progress = (current_step * 100) / total_steps;
        cout << "Progress: " << progress << "%" << endl;
    }
}


// Return the input vector train and prediction subsets. Do not change input vector.
void testTrainSplit(
    const Eigen::MatrixXd& input_matrix,
    int train_length,
    int prediction_length,
    Eigen::MatrixXd& train_matrix,
    Eigen::MatrixXd& predict_matrix,
    int skip_n) {

    // Handle empty matrix (e.g. no forcing function).
    if (input_matrix.rows() == 0 || input_matrix.cols() == 0) {
        train_matrix = Eigen::MatrixXd(0, 0);
        predict_matrix = Eigen::MatrixXd(0, 0);
        return;
    }

    // Get the number of rows (each time series stored as row.).
    int num_rows = input_matrix.rows();

    // Check lengths are sensible.
    if (skip_n + train_length + prediction_length > input_matrix.cols()) {
        cerr << "Error: Combined analysis window larger than dataset." << endl;
        exit(0);
    }

    train_matrix = input_matrix.block(0, skip_n, num_rows, train_length);
    predict_matrix = input_matrix.block(0, skip_n + train_length, num_rows, prediction_length);
}


// ----------------------------------------------------------------------------------------------------


// Function to write data to a CSV file
void writeVectorofVectorToCSV(string& filename, vector<vector<double>>& output_data, string& header) {
    // Basic CSV writer.

    cout << "Exporting results to CSV" << endl;

    ofstream csvFile(filename);

    if (csvFile.is_open()) {
        csvFile << header << "\n";

        for (int i = 0; i < output_data.size(); i++) {
            csvFile << i << "," << fixed << setprecision(6); // Control precision (many double types)

            for (int j = 0; j < output_data[i].size(); j++) {
                csvFile << output_data[i][j];

                if (j < output_data[i].size() - 1) csvFile << ",";
            }
            csvFile << "\n";
        }
        csvFile.close();
        cout << "Results written to " << filename << endl;
    }
    else {
        cout << "Unable to open csvFile." << endl;
        exit(0);
    }
};


void writeMatrixToCSV(const std::string& filename, const Eigen::MatrixXd& matrix) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            outFile << matrix(i, j);
            if (j < matrix.cols() - 1) {
                outFile << ","; // Add comma between values
            }
        }
        outFile << "\n"; // New line after each row
    }

    outFile.close(); // Close the file
}


void exportPredictionsToCSV(
    const Eigen::MatrixXd& matrix1,
    const Eigen::MatrixXd& matrix2,
    const string& filename,
    const string& prefix1, 
    const string& prefix2   
) {
    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(0);
    }


    // Write headers
    for (int i = 0; i < matrix1.rows(); ++i) {
        file << prefix1 << " " << (i + 1) << ",";
    }

    file << " "; // Blank column

    for (int i = 0; i < matrix2.rows(); ++i) {
        file << prefix2 << " " << (i + 1) << ",";
    }
    file << endl;


    // Write data
    for (int j = 0; j < matrix1.cols(); ++j) {

        for (int i = 0; i < matrix1.rows(); ++i) {
            file << matrix1(i, j) << ",";
        }

        file << " "; // Blank col

        for (int i = 0; i < matrix2.rows(); ++i) {
            file << matrix2(i, j) << ",";
        }
        file << endl;
    }

    file.close();
    cout << "Results written to " << filename << endl;
}


void logGridSearchTrialParams(
    const string& filename,
    int trial_number,
    const RC_ParamStruct& params,
    double trial_mae,
    double trial_rmse
) {
    
    // Open in append mode.
    ofstream file;
    file.open(filename, ios_base::app);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(0);
    }

    // Header on first.
    if (trial_number == 1) {
        file << "Trial Number,Spectral Radius,Degree,Sigma,Beta,Approx Size,Train Length,Prediction Length,Skip Length,Scaling,MAE,RMSE\n";
    }

    // Write the current trial data
    file << trial_number << ", "
        << params.spectral_radius << ", "
        << params.degree << ", "
        << params.sigma << ", "
        << params.beta << ", "
        << params.approx_size << ", "
        << params.train_length << ", "
        << params.predict_length << ", "
        << params.skip_length << ", "
        << params.scaling << ", "
        << trial_mae<< ", "
        << trial_rmse << "\n";


    file.close();
}


void logGenericTrial(
    const string& filename,
    string header,
    int trial_number,
    string trial_row
) {

    // Open in append mode.
    ofstream file;
    file.open(filename, ios_base::app);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(0);
    }

    // Header on first.
    if (trial_number == 1) { file << header << "\n"; }

    // Write the current trial data
    file << trial_row << "\n";


    file.close();
}

// ----------------------------------------------------------------------------------------------------


void applyGaussianNoise(Eigen::MatrixXd& matrix, double std_noise, int rng_seed) {

    // RNG seeded for reproducibility. 
    mt19937 gen(rng_seed);
    normal_distribution<double> distribution(0.0, std_noise);

    // Add noise elementwise on matrix.
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            matrix(i, j) += distribution(gen);
        }
    }
}


// ----------------------------------------------------------------------------------------------------
// Scaler.

// Fit scaler to training data and store results as private attributes.
void Scaler::fit(const Eigen::MatrixXd& data) {
    
    mean.resize(data.rows());
    std_dev.resize(data.rows());

    // Calculate mean and std deviation for each row
    for (int i = 0; i < data.rows(); ++i) {
        mean(i) = data.row(i).mean();
        std_dev(i) = sqrt((data.row(i).array() - mean(i)).square().mean());
    }
}

// Scale the data
Eigen::MatrixXd Scaler::scale(const Eigen::MatrixXd& data) const {
    
    Eigen::MatrixXd scaledData = data;

    for (int i = 0; i < data.rows(); ++i) {
        
        if (std_dev(i) == 0) {
            cerr << "STDev of zero, cannot scale." << endl;
            exit(0);
        }

        // Z Scaling
        scaledData.row(i) = (data.row(i).array() - mean(i)) / std_dev(i);
    }

    return scaledData;
}

// Unscale the data
Eigen::MatrixXd Scaler::unscale(const Eigen::MatrixXd& scaledData) const {
    
    Eigen::MatrixXd originalData = scaledData;

    for (int i = 0; i < scaledData.rows(); ++i) {
        originalData.row(i) = (scaledData.row(i).array() * std_dev(i)) + mean(i);
    }

    return originalData;
}