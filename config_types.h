#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <string>

using namespace std;

// ----------------------------------------------------------------------------------------------------

struct HH_ParamStruct
{

	// Export filepath.
	string export_filepath;

	// Stim Current parameters
	vector<string> stimFilePaths;
	double stimScale;
	int samplingInterval;

	// Constants
	double TEMP_C; // Temperature

	// Model parameters
	double T;  // Model duration in ms
	double dt; // Time step in ms

	// Neuron parameters
	double soma_len;  // Soma length in cm
	double soma_diam; // Soma diameter in cm
	double Cm;		  // Membrane capacitance

	// Channel parameters
	double gNaT;  // Sodium channel conductance
	double ENa;	  // Sodium reversal potential
	double gK;	  // Potassium channel conductance
	double EK;	  // Potassium reversal potential
	double gLeak; // Leak conductance
	double EL;	  // Leak reversal potential

	// Gating variable parameters
	double amV1, amV2, amV3; // parameters for m-gate
	double tm0, epsm;
	double ahV1, ahV2, ahV3; // parameters for h-gate
	double th0, epsh;
	double anV1, anV2, anV3; // parameters for n-gate
	double tn0, epsn;

	// Initial conditions
	double V_init; // Initial membrane potential
	double m_init; // Initial m-gate value
	double h_init; // Initial h-gate value
	double n_init; // Initial n-gate value
};

struct Lorenz_ParamStruct
{

	// Export filepath.
	string export_filepath;

	// Model constants.
	double sigma;
	double rho;
	double beta;

	// Model parameters
	double T;  // Model duration in ms
	double dt; // Time step in ms

	// Initial conditions.
	double x_init;
	double y_init;
	double z_init;
};

struct RC_ParamStruct
{

	double spectral_radius; // Stability
	double degree;			// Connectivity
	double sigma;			// Input responsiveness.
	double beta;			// Ridge regression regularisation.
	int approx_size;		// Approximate size of reservoir.

	int train_length;	// Training Steps
	int predict_length; // Prediction Steps
	int skip_length;	// Skip n entries at start of timeseries. (default 0)

	bool scaling; // Whether to apply input scaling.

	bool input_hybrid = true;  // If a KBM exists, control hybrid connections
	bool output_hybrid = true; // If a KBM exists, control hybrid connections
};

// ----------------------------------------------------------------------------------------------------