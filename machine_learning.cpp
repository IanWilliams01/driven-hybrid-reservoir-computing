#pragma once

#include "kbm_wrapper.h"
#include "config_types.h"
#include "utils.h"

#define _USE_MATH_DEFINES
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>

#include <armadillo>
#include <random>

extern bool debug_outputs;
using namespace std;

// ----------------------------------------------------------------------------------------------------


class ReservoirComputing {
private:

	// KBM Pointer
	kbmWrapper* KBMInstance = nullptr;

	// Training results
	Eigen::SparseMatrix<double> adjacency_matrix;
	Eigen::MatrixXd input_weights;
	Eigen::MatrixXd output_weights;
	Eigen::VectorXd final_reservoir_state;


public:

	// ----------------------------------------------------------------------------------------------------
	// Constants

	int rng_seed = 42;

	// ----------------------------------------------------------------------------------------------------
	// Parameters

	double spectral_radius;		// Stability
	double degree;				// Connectivity
	double sigma;				// Input responsiveness.
	double beta;				// Ridge regression regularisation.

	int L;						// L (Num of dependent input feeds)
	int F;						// F (Num of forcing functions) 

	int N;						// Nodes in reservoir.
	int num_inputs;				// Total input streams.

	int train_length;			// Training Steps
	int predict_length;			// Prediction Steps
	int skip_length;			// Steps to skip at start

	bool scaling;				// Whether to apply input Z-Scaling

	bool input_hybrid;			// KBM Controls.
	bool output_hybrid;

	// ----------------------------------------------------------------------------------------------------
	// Pre-Processing & Scaling.

	Scaler dataScaler;
	Scaler forcScaler;
	Scaler kbmScaler;
	
	// ----------------------------------------------------------------------------------------------------
	// Time-Series & Metrics

	Eigen::MatrixXd data_full;
	Eigen::MatrixXd data_train;
	Eigen::MatrixXd data_predict;

	Eigen::MatrixXd forc_full;
	Eigen::MatrixXd forc_train;
	Eigen::MatrixXd forc_predict;

	Eigen::MatrixXd kbm_full;
	Eigen::MatrixXd kbm_train;
	Eigen::MatrixXd kbm_predict; 

	Eigen::MatrixXd predictions;
	double predictions_mae;
	double predictions_rmse;


	// ----------------------------------------------------------------------------------------------------
	// Constructor & Destructor

	ReservoirComputing(const RC_ParamStruct& configSelected, Eigen::MatrixXd inputData, Eigen::MatrixXd inputForc = Eigen::MatrixXd::Zero(0, 0), kbmWrapper* KBM_ptr = nullptr) {

		cout << "\nML: Reservoir Computing System -----------------" << endl;

		// Unpack the config struct.
		spectral_radius = configSelected.spectral_radius;
		degree = configSelected.degree;
		sigma = configSelected.sigma;
		beta = configSelected.beta;

		train_length = configSelected.train_length;
		predict_length = configSelected.predict_length;
		skip_length = configSelected.skip_length;

		scaling = configSelected.scaling;

		input_hybrid = configSelected.input_hybrid;
		output_hybrid = configSelected.output_hybrid;


		// Unpack the time-series data.
		data_full = inputData;
		testTrainSplit(data_full, train_length, predict_length, data_train, data_predict, skip_length);
		forc_full = inputForc;
		testTrainSplit(forc_full, train_length, predict_length, forc_train, forc_predict, skip_length);

		// Run KBM if provided.
		KBMInstance = KBM_ptr;
		if (KBMInstance) {
			kbm_full = KBMInstance->generateKBMfull(data_full, forc_full); 
			testTrainSplit(kbm_full, train_length, predict_length, kbm_train, kbm_predict, skip_length);
		}
		else
		{
			kbm_full = Eigen::MatrixXd::Zero(0, 0);
		}
		

		// Debug Outputs.
		if (debug_outputs) { 
			writeMatrixToCSV("output/data_full.csv", data_full);
			writeMatrixToCSV("output/data_train.csv", data_train);
			writeMatrixToCSV("output/data_predict.csv", data_predict);
			writeMatrixToCSV("output/forc_full.csv", forc_full);
			writeMatrixToCSV("output/forc_train.csv", forc_train);
			writeMatrixToCSV("output/forc_predict.csv", forc_predict);
		}


		// Determine number of input & forcing time-series. 
		L = inputData.rows();
		F = inputForc.rows();
		if (input_hybrid) { F = F + kbm_full.rows(); }
		cout << "RC configured for L: " << L << ", F: " << F << endl;


		// Use to calculate the size of the reservoir (adjusted if required to divide nicely).
		num_inputs = (L + F);
		N = static_cast<int>(
			floor(
				static_cast<double>(configSelected.approx_size) / num_inputs
			) * num_inputs
			);


		// Apply scaling.
		if (scaling) { applyScaling(); }

	};
	~ReservoirComputing() = default;


	// ----------------------------------------------------------------------------------------------------
	// Pre-Processing & Scaling.

	void applyScaling() {

		cout << "Applying input scaling" << endl;

		// Fit on training data (to stop forward-bias) and scale all.
		dataScaler.fit(data_train);
		data_full = dataScaler.scale(data_full);
		data_train = dataScaler.scale(data_train);
		data_predict = dataScaler.scale(data_predict);
		
		if (forc_full.rows() != 0 && forc_full.cols() != 0) {
			forcScaler.fit(forc_train);
			forc_full = forcScaler.scale(forc_full);
			forc_train = forcScaler.scale(forc_train);
			forc_predict = forcScaler.scale(forc_predict);
		}
	}

	void undoScaling() {

		cout << "Rescaling all time-series" << endl;

		data_full = dataScaler.unscale(data_full);
		data_train = dataScaler.unscale(data_train);
		data_predict = dataScaler.unscale(data_predict);
		predictions = dataScaler.unscale(predictions);

		if (forc_full.rows() != 0 && forc_full.cols() != 0) {
			forc_full = forcScaler.unscale(forc_full);
			forc_train = forcScaler.unscale(forc_train);
			forc_predict = forcScaler.unscale(forc_predict);
		}
	}


	// ----------------------------------------------------------------------------------------------------
	// Reservoir Computing Functionality


	Eigen::SparseMatrix<double> initialiseReservoir() {

		cout << "Initialising reservoir of size " << N << endl;

		// RNG seeded for reproducibility. 
		mt19937 gen(rng_seed);
		uniform_real_distribution<double> distribution(0.0, 1.0);

		// Sparsity calculations
		double sparsity = degree / N;
		int nonZeros = static_cast<int>(sparsity * N * N);

		// Create a sparse matrix
		Eigen::SparseMatrix<double> adjacency_matrix(N, N);

		// Set values in A using random vals fron uniform dist.
		// (random locations, number of non-zero controlled above by sparsity).
		for (int i = 0; i < nonZeros; ++i) {

			// Random row/col indices
			int row = gen() % N;
			int col = gen() % N;

			// Set random value [0, 1) in sparse matrix
			adjacency_matrix.coeffRef(row, col) = distribution(gen);
		}

		// ----- Calculate eigenvalues using Armadillo ----------------

		// Copy Eigen sparse matrix to Armadillo sparse matrix (faster to find eigenvals)
		arma::sp_mat arma_sparse(adjacency_matrix.rows(), adjacency_matrix.cols());
		for (int k = 0; k < adjacency_matrix.outerSize(); ++k) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_matrix, k); it; ++it) {
				arma_sparse(it.row(), it.col()) = it.value();
			}
		}
		// Compute the max eigenvalues (eigs_gen details can be seen in armadillo docs)
		arma::Col<complex<double>> eigvals;
		arma::eigs_gen(eigvals, arma_sparse, 1, "lm");
		double max_eigenval = abs(real(eigvals(0)));

		// ------------------------------------------------------------

		// Rescale on spectral rad
		adjacency_matrix = (adjacency_matrix / max_eigenval) * spectral_radius;

		return adjacency_matrix;

	};


	Eigen::MatrixXd reservoirLayer(Eigen::MatrixXd& data, Eigen::MatrixXd& forc, Eigen::MatrixXd& kbm) {
		
		cout << "Calculating states across training window" << endl;

		// Create and fill matrix combining data & forc. Each col is a timestep.
		Eigen::MatrixXd input;

		// Convoluted conditional to determine handle all types of input structures.
		if (forc.rows() == 0 && forc.cols() == 0 && kbm.rows() == 0 && kbm.cols() == 0) {
			input = data;
		}
		else if (kbm.rows() == 0 && kbm.cols() == 0) {
			input.resize(data.rows() + forc.rows(), data.cols());
			input << data, forc;
		}
		else if (forc.rows() == 0 && forc.cols() == 0) {

			if (input_hybrid) {
				input.resize(data.rows() + kbm.rows(), data.cols());
				input << data, kbm;
			}
			else {
				input = data;
			}
		}
		else {
			
			if (input_hybrid) {
				input.resize(data.rows() + forc.rows() + kbm.rows(), data.cols());
				input << data, forc, kbm;
			}
			else {
				input.resize(data.rows() + forc.rows(), data.cols());
				input << data, forc;
			}
			
		}


		// Debug Outputs.
		if (debug_outputs) {
			writeMatrixToCSV("output/training_input.csv", input);
		}


		// Create matrix of zeros to hold states. (filled by reservoir dynamics)
		Eigen::MatrixXd states = Eigen::MatrixXd::Zero(N, train_length);

		// Calculate states across *training* window.
		for (int i = 0; i < train_length - 1; ++i) {

			progressBar(i, train_length);

			states.col(i + 1) = (adjacency_matrix * states.col(i)).array() + (input_weights * input.col(i)).array();
			states.col(i + 1) = states.col(i + 1).array().tanh();

		}

		cout << "States: " << states.rows() << " x " << states.cols() << endl;


		// "fake" states appended to bottom (acting as pipeline for kbm into output layer).
		// Allows for output layer weights to be calibrated using existing logic.
		if (kbm.rows() != 0 && kbm.cols() != 0 && output_hybrid) {

			cout << "Adding 'fake' KBM states to output layer" << endl;

			if (scaling) {
				cout << "Scaling KBM" << endl;
				kbmScaler.fit(kbm);
				kbm = kbmScaler.scale(kbm);
			}

			// TODO: CLARITY OF EXPLANATION.
			// Shift to the right by 1 index to attach states in right place.
			// Careful here. Note that above is setting i+1 state using input at i across training window.
			// i+1 state x output weights = i+1 prediction.
			// Therefore, "fake states" should correspond to the KBM pred value at i+1.
			// Hence, the right shift by one index.
			Eigen::MatrixXd shifted_kbm = kbm;
			shifted_kbm.rightCols(kbm.cols() - 1) = kbm.leftCols(kbm.cols() - 1);
			shifted_kbm.col(0).setZero();
			kbm = shifted_kbm;

			// Combine
			Eigen::MatrixXd combined_states(states.rows() + kbm.rows(), states.cols());
			combined_states.topRows(states.rows()) = states;
			combined_states.bottomRows(kbm.rows()) = kbm;


			states = combined_states;
		}

		cout << "States: " << states.rows() << " x " << states.cols() << endl;
		cout << "Progress: 100%" << endl;

		return states;
	};


	Eigen::MatrixXd calibrateOutputWeights(Eigen::MatrixXd& states, Eigen::MatrixXd& data) {
		cout << "Optimising output weights" << endl;

		// Copy to modify later.
		Eigen::MatrixXd modifiedStates = states;


		// Debug Outputs.
		if (debug_outputs) {
			// Commented out as very large files, slow to export.
			//writeMatrixToCSV("output/calibrateOutputWeights_states.csv", states);
			//writeMatrixToCSV("output/calibrateOutputWeights_data.csv", data);
		}


		// ----- Calculate covariance matrix & output weights using Armadillo ----------------

		arma::mat modifiedStatesArma = arma::mat(modifiedStates.data(), modifiedStates.rows(), modifiedStates.cols(), false, true); 
		arma::mat idenmatArma = beta * arma::eye<arma::mat>(modifiedStatesArma.n_rows, modifiedStatesArma.n_rows);

		arma::mat covariance_matrix = modifiedStatesArma * modifiedStatesArma.t() + idenmatArma;
		arma::mat covariance_matrix_inv = arma::inv(covariance_matrix);

		// Compute output layer weights 
		Eigen::MatrixXd output_weights = Eigen::Map<Eigen::MatrixXd>(covariance_matrix_inv.memptr(), covariance_matrix_inv.n_rows, covariance_matrix_inv.n_cols) * (modifiedStates * data.transpose());

		cout << "Optimisation complete" << endl;

		return output_weights.transpose();

	};


	// ------------------------------------------------------------


	void trainReservoir() {

		cout << "\nTraining reservoir computer" << endl;

		// RNG seeded for reproducibility.
		mt19937 gen(rng_seed);
		uniform_real_distribution<double> distribution(-1.0, 1.0);

		// Handle defined KBM Input Fraction
		if (KBMInstance && KBMInstance->kbm_input_fraction != 0) {

			cout << "Applying custom KBM input fraction" << endl;

			float kbm_fraction = KBMInstance->kbm_input_fraction;

			int num_kbm = kbm_full.rows();
			int num_normal = data_full.rows() + forc_full.rows();

			int connectionsPerKBM = static_cast<int>((N * kbm_fraction) / num_kbm);
			int connectionsPerNonKBM = static_cast<int>((N * (1 - kbm_fraction)) / num_normal);

			// Modify reservoir size
			N = (connectionsPerKBM * num_kbm) + (connectionsPerNonKBM * num_normal);
			cout << "Modified reservoir for custom input fraction." << endl;
			cout << "New: " << N << endl;

			cout << "connectionsPerKBM: " << connectionsPerKBM << endl;
			cout << "connectionsPerNonKBM: " << connectionsPerNonKBM << endl;


			// Generate initial adjacency matrix
			adjacency_matrix = initialiseReservoir();

			// Input input weights matrix of zeros (to be filled).
			// Note, stored as attribute, later accessed by reservoirLayer.
			input_weights = Eigen::MatrixXd::Zero(N, num_inputs);

			// Assign Normal fraction
			for (int i = 0; i < num_normal; ++i) {
				for (int j = 0; j < connectionsPerNonKBM; ++j) {
					// Sigma scaling (Input responsiveness for normal inputs
					//cout << (i * connectionsPerNonKBM + j) << " , " << (i) << endl;
					input_weights(i * connectionsPerNonKBM + j, i) = sigma * distribution(gen);
				}
			}

			// Assign KBM fraction. (note KBM rows are last in the input object)
			for (int i = 0; i < num_kbm; ++i) {
				for (int j = 0; j < connectionsPerKBM; ++j) {
					// Sigma scaling (Input responsiveness for KBM inputs)
					//cout << ((num_normal * connectionsPerNonKBM) + i * connectionsPerKBM + j) << " , " << (num_normal + i) << endl;
					input_weights((num_normal * connectionsPerNonKBM) + i * connectionsPerKBM + j, num_normal + i) = sigma * distribution(gen);
				}
			}

			// Debug Outputs.
			if (debug_outputs) {
				writeMatrixToCSV("output/input_weights_modified_kbm.csv", input_weights);
			}

		}
		// Equally weighted.
		else
		{
			int connectionsPerInput = static_cast<int>(N / num_inputs);

			// Generate initial adjacency matrix
			adjacency_matrix = initialiseReservoir();

			// Input input weights matrix of zeros (to be filled).
			// Note, stored as attribute, later accessed by reservoirLayer.
			input_weights = Eigen::MatrixXd::Zero(N, num_inputs);

			for (int i = 0; i < num_inputs; ++i) {
				for (int j = 0; j < connectionsPerInput; ++j) {
					// Sigma scaling (Input responsiveness)
					input_weights(i * connectionsPerInput + j, i) = sigma * distribution(gen);
				}
			}

			// Debug Outputs.
			if (debug_outputs) {
				writeMatrixToCSV("output/input_weights.csv", input_weights);
			}
		}

		// Calculate the reservoir states using reservoirLayer()
		Eigen::MatrixXd states = reservoirLayer(data_train, forc_train, kbm_train);

		// Train output weights states and data
		output_weights = calibrateOutputWeights(states, data_train);

		// Debug Outputs.
		if (debug_outputs) {
			writeMatrixToCSV("output/output_weights.csv", output_weights);
		}

		// Store last column (final state).
		final_reservoir_state = states.col(states.cols() - 1);

	};


	void generatePredictions() {

		cout << "\nGenerating predictions" << endl;

		// Initialize predicted output matrix as zeros
		Eigen::MatrixXd predicted_output = Eigen::MatrixXd::Zero(L, predict_length);

		// Initial prediction for final state from training. 
		// Careful here. This prediction is used to propogate the reservoir state forward into
		// the prediction window. This prediction itself is part of the training set.
		Eigen::VectorXd reservoir_state = final_reservoir_state;
		Eigen::VectorXd prediction = output_weights * reservoir_state;


		// Iteratively predict future states
		for (int time_step = 0; time_step < predict_length; ++time_step) {

			progressBar(time_step, predict_length);

			Eigen::MatrixXd kbm_tplus1;

			// KBM Specific logic.
			if (KBMInstance) {

				if (scaling) {
					Eigen::MatrixXd unscaled_prediction = dataScaler.unscale(prediction);
					kbm_tplus1 = KBMInstance->generateKBMlive(unscaled_prediction, time_step, skip_length, train_length);
					kbm_tplus1 = kbmScaler.scale(kbm_tplus1);
				}
				else
				{
					kbm_tplus1 = KBMInstance->generateKBMlive(prediction, time_step, skip_length, train_length);
				}

				// remove "fake" rows for calculating new reservoir state.
				if (output_hybrid) {
					int rows_to_keep = reservoir_state.rows() - kbm_tplus1.rows();
					reservoir_state.conservativeResize(rows_to_keep, reservoir_state.cols());
				}
			}


			// Convoluted conditional to determine handle all types of input structures.
			Eigen::MatrixXd combined_input(num_inputs, 1);
			if (forc_predict.rows() == 0 && forc_predict.cols() == 0 && kbm_tplus1.rows() == 0 && kbm_tplus1.cols() == 0) {
				combined_input << prediction;
			}
			else if (kbm_tplus1.rows() == 0 && kbm_tplus1.cols() == 0) {
				combined_input << prediction, forc_predict.col(time_step); 
			}
			else if (forc_predict.rows() == 0 && forc_predict.cols() == 0) {

				if (input_hybrid) {
					combined_input << prediction, kbm_tplus1.col(0);
				}
				else {
					combined_input << prediction;
				}

			}
			else {

				if (input_hybrid) {
					combined_input << prediction, forc_predict.col(time_step), kbm_tplus1.col(0);
				}
				else {
					combined_input << prediction, forc_predict.col(time_step);
				}

			}


			// Calculate new reservoir state
			Eigen::VectorXd new_reservoir_state = (adjacency_matrix * reservoir_state).array() +
				(input_weights * combined_input).array();
			new_reservoir_state = new_reservoir_state.array().tanh();


			//// Re-add "fake" kbm rows for next prediction.
			if (kbm_tplus1.rows() != 0 && kbm_tplus1.cols() != 0 && output_hybrid) {
				Eigen::MatrixXd combined_states(new_reservoir_state.rows() + kbm_tplus1.rows(), new_reservoir_state.cols());
				combined_states.topRows(new_reservoir_state.rows()) = new_reservoir_state;
				combined_states.bottomRows(kbm_tplus1.rows()) = kbm_tplus1;
				new_reservoir_state = combined_states;
			}

			// Generate prediction for this timestep
			prediction = output_weights * new_reservoir_state;
			predicted_output.col(time_step) = prediction;


			// Update final reservoir state for the next iteration
			reservoir_state = new_reservoir_state;

		}

		cout << "Progress: 100% \n" << endl;
		predictions = predicted_output;

		// Undo Scaling if applied earlier (see utils Scaler).
		if (scaling) { undoScaling(); }

		// Assess performance
		computePredictionMAE();
		computePredictionRMSE();
	};


	// ------------------------------------------------------------


	void computePredictionMAE() {
		cout << "Assessing performance (MAE) " << endl;

		Eigen::MatrixXd abs_diff = (data_predict - predictions).cwiseAbs();
		predictions_mae = abs_diff.mean();

		cout << "MAE: " << to_string(predictions_mae) << endl;
	}

	void computePredictionRMSE() {
		cout << "Assessing performance (RMSE) " << endl;

		Eigen::MatrixXd squared_diff = (data_predict - predictions).array().square();
		predictions_rmse = sqrt(squared_diff.mean());

		cout << "RMSE: " << to_string(predictions_rmse) << endl;
	}

};


