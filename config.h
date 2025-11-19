#pragma once

#include "config_types.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <string>

using namespace std;

// ----------------------------------------------------------------------------------------------------
// Global Config Switches

bool debug_outputs = false;

// ----------------------------------------------------------------------------------------------------
// Diagnostic

HH_ParamStruct HH_diagnostic = {

	// Export filepath.
	"output/HH_diagnostic.csv",

	// Stim Current parameters
	{"stim_files/Pstandard_100khz_0.dat",
	 "stim_files/Pstandard_100khz_1.dat",
	 "stim_files/Pstandard_100khz_2.dat",
	 "stim_files/Pstandard_100khz_3.dat",
	 "stim_files/Pstandard_100khz_4.dat"},
	0.02, // stimScale
	20,	  // samplingInterval

	// Constants
	35, // Temperature

	// Model parameters
	7400,  // Model duration (ms)
	0.025, // Time step (ms)

	// Neuron parameters
	0.01,		  // Soma length (cm)
	0.029 / M_PI, // Soma diameter (cm)
	1,			  // Membrane capacitance

	// Channel parameters
	69,	   // Sodium channel conductance
	41,	   // Sodium reversal potential
	6.9,   // Potassium channel conductance
	-100,  // Potassium reversal potential
	0.465, // Leak conductance
	-65,   // Leak reversal potential

	// Gating variable parameters
	-39.92,
	10,
	23.39, // parameters for m-gate
	0.143,
	1.099,
	-65.37,
	-17.65,
	27.22, // parameters for h-gate
	0.701,
	12.90,
	-34.58,
	22.17,
	23.58, // parameters for n-gate
	1.291,
	4.314,

	// Initial conditions
	-65,	 // Initial membrane potential
	0.00742, // Initial m-gate value
	0.47258, // Initial h-gate value
	0.06356	 // Initial n-gate value
};

Lorenz_ParamStruct LZ_diagnostic = {

	// Export filepath.
	"output/Lorenz_diagnostic.csv",

	// Model constants.
	10,
	28,
	8 / 3,

	// Model parameters.
	10000, // Model duration in ms
	0.01,  // Time step in ms

	// Initial conditions.
	2,
	1,
	1,
};

RC_ParamStruct RC_diagnostic = {
	1.25,	// spectral_radius (Stability)
	6,		// degree (Connectivity)
	0.4,	// sigma (Input responsiveness.)
	0.0001, // beta (Ridge regression regularisation)
	10,		// approx_size (Approximate size of reservoir.)

	5000, // train_length;
	500,  // predict_length;
	0,	  // skip_length;

	true, // scaling;

};

// ----------------------------------------------------------------------------------------------------

RC_ParamStruct RC_CoLab_Legacy = {
	1.25,	// spectral_radius (Stability)
	6,		// degree (Connectivity)
	0.4,	// sigma (Input responsiveness.)
	0.0001, // beta (Ridge regression regularisation)
	2000,	// approx_size (Approximate size of reservoir.)

	100000, // train_length;
	50000,	// predict_length;
	0,		// skip_length;

	true, // scaling
};

RC_ParamStruct RC_trial_LZ_Test = {
	1.25,	// spectral_radius (Stability)
	6,		// degree (Connectivity)
	0.4,	// sigma (Input responsiveness.)
	0.0001, // beta (Ridge regression regularisation)
	500,	// approx_size (Approximate size of reservoir.)

	5000, // train_length;
	500,  // predict_length;
	0,	  // skip_length;

	false, // scaling;
};

RC_ParamStruct RC_trial_HH_Test = {
	1.25,	// spectral_radius (Stability)
	6,		// degree (Connectivity)
	0.5,	// sigma (Input responsiveness.)
	0.0001, // beta (Ridge regression regularisation)
	2000,	// approx_size (Approximate size of reservoir.)

	50000, // train_length;
	50000, // predict_length;
	0,	   // skip_length;

	false, // scaling;
};

RC_ParamStruct RC_trial_HH_adjopt = {
	1.25,  // spectral_radius (Stability)
	8,	   // degree (Connectivity)
	0.65,  // sigma (Input responsiveness.)
	0.001, // beta (Ridge regression regularisation)
	2000,  // approx_size (Approximate size of reservoir.)

	100000, // train_length;
	50000,	// predict_length;
	0,		// skip_length;

	false, // scaling;

};

RC_ParamStruct RC_trial_HH_multi = {
	1.25,  // spectral_radius (Stability)
	8,	   // degree (Connectivity)
	0.35,  // sigma (Input responsiveness.)
	0.001, // beta (Ridge regression regularisation)
	2000,  // approx_size (Approximate size of reservoir.)

	100000, // train_length;
	50000,	// predict_length;
	0,		// skip_length;

	false, // scaling;

};

// ----------------------------------------------------------------------------------------------------
// KBM - LZ

Lorenz_ParamStruct lorenz_trial = {

	// Export filepath.
	"output/lorenz_trial.csv",

	// Model constants.
	10,
	28,
	8 / 3,

	// Model parameters.
	10000, // Model duration in ms
	0.1,   // Time step in ms

	// Initial conditions.
	2,
	1,
	1,
};

Lorenz_ParamStruct lorenz_trial_kbm = {

	// Export filepath.
	"output/lorenz_trial_kbm.csv",

	// Model constants.
	10,
	28 * (1.05),
	8 / 3,

	// Model parameters.
	10000, // Model duration in ms
	0.1,   // Time step in ms

	// Initial conditions.
	2,
	1,
	1,
};

RC_ParamStruct RC_trial_LZ_kbm_Test = {
	0.4,	  // spectral_radius (Stability)
	3,		  // degree (Connectivity)
	0.3,	  // sigma (Input responsiveness.)
	0.000001, // beta (Ridge regression regularisation)
	500,	  // approx_size (Approximate size of reservoir.)

	2000, // train_length;
	250,  // predict_length;
	5000, // skip_length;

	false, // scaling;
};

// ----------------------------------------------------------------------------------------------------
// KBM - HH

HH_ParamStruct HH_trial = {

	// Export filepath.
	"output/HH_trial.csv",

	// Stim Current parameters
	{"stim_files/Pstandard_100khz_0.dat",
	 "stim_files/Pstandard_100khz_1.dat",
	 "stim_files/Pstandard_100khz_2.dat",
	 "stim_files/Pstandard_100khz_3.dat",
	 "stim_files/Pstandard_100khz_4.dat"},
	0.02, // stimScale
	20,	  // samplingInterval

	// Constants
	35, // Temperature

	// Model parameters
	7400,  // Model duration (ms)
	0.025, // Time step (ms)

	// Neuron parameters
	0.01,		  // Soma length (cm)
	0.029 / M_PI, // Soma diameter (cm)
	1,			  // Membrane capacitance

	// Channel parameters
	69,	   // Sodium channel conductance
	41,	   // Sodium reversal potential
	6.9,   // Potassium channel conductance
	-100,  // Potassium reversal potential
	0.465, // Leak conductance
	-65,   // Leak reversal potential

	// Gating variable parameters
	-39.92,
	10,
	23.39, // parameters for m-gate
	0.143,
	1.099,
	-65.37,
	-17.65,
	27.22, // parameters for h-gate
	0.701,
	12.90,
	-34.58,
	22.17,
	23.58, // parameters for n-gate
	1.291,
	4.314,

	// Initial conditions
	-65,	 // Initial membrane potential
	0.00742, // Initial m-gate value
	0.47258, // Initial h-gate value
	0.06356	 // Initial n-gate value
};

HH_ParamStruct HH_trial_kbm = {

	// Export filepath.
	"output/HH_trial_kbm.csv",

	// Stim Current parameters
	{"stim_files/Pstandard_100khz_0.dat",
	 "stim_files/Pstandard_100khz_1.dat",
	 "stim_files/Pstandard_100khz_2.dat",
	 "stim_files/Pstandard_100khz_3.dat",
	 "stim_files/Pstandard_100khz_4.dat"},
	0.02, // stimScale
	20,	  // samplingInterval

	// Constants
	35, // Temperature

	// Model parameters
	7400,  // Model duration (ms)
	0.025, // Time step (ms)

	// Neuron parameters
	0.01,		  // Soma length (cm)
	0.029 / M_PI, // Soma diameter (cm)
	1,			  // Membrane capacitance

	// Channel parameters
	69,	   // Sodium channel conductance
	41,	   // Sodium reversal potential
	6.9,   // Potassium channel conductance
	-100,  // Potassium reversal potential
	0.465, // Leak conductance
	-65,   // Leak reversal potential

	// Gating variable parameters
	-39.92,
	10,
	23.39, // parameters for m-gate
	0.143,
	1.099,
	-65.37,
	-17.65,
	27.22, // parameters for h-gate
	0.701,
	12.90,
	-34.58,
	22.17,
	23.58, // parameters for n-gate
	1.291,
	4.314,

	// Initial conditions
	-65,	 // Initial membrane potential
	0.00742, // Initial m-gate value
	0.47258, // Initial h-gate value
	0.06356	 // Initial n-gate value
};

RC_ParamStruct RC_trial_HH_kbm_test = {
	1,	   // spectral_radius (Stability)
	8,	   // degree (Connectivity)
	0.8,   // sigma (Input responsiveness.)
	0.001, // beta (Ridge regression regularisation)
	1000,  // approx_size (Approximate size of reservoir.)

	50000, // train_length;
	10000, // predict_length;
	0,	   // skip_length;

	false, // scaling;
};

RC_ParamStruct RC_hybrid_internal_variables = {
	1,	   // spectral_radius (Stability)
	8,	   // degree (Connectivity)
	0.8,   // sigma (Input responsiveness.)
	0.001, // beta (Ridge regression regularisation)
	1000,  // approx_size (Approximate size of reservoir.)

	100000, // train_length;
	50000,	// predict_length;
	0,		// skip_length;

	false, // scaling;
};