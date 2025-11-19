#ifndef HODGKIN_HUXLEY_H
#define HODGKIN_HUXLEY_H

#include "config_types.h"
#include "utils.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;
using namespace std;


class HodgkinHuxley {
private:
public:
	// ----------------------------------------------------------------------------------------------------
	// Parameters

	// Runge-Kutta Cash-Karp 5(4) for integration
	runge_kutta_cash_karp54<vector<double>> rkck;

	// Export filepath.
	string export_filepath;

	// Stim Current parameters
	vector<string> stimFilePaths;
	double stimScale;
	int samplingInterval;
	vector<double> stim; // Resultant external stimulation array

	// Constants
	double TEMP_C;		// Temperature
	double FARADAY = 96480;
	double PI = M_PI;

	// Model parameters
	double T;			// Model duration in ms
	double dt;			// Time step in ms
	int num_steps;

	// Neuron parameters
	double soma_len;	// Soma length in cm
	double soma_diam;	// Soma diameter in cm
	double Cm;			// Membrane capacitance

	// Channel parameters
	double gNaT;		// Sodium channel conductance
	double ENa;			// Sodium reversal potential
	double gK;			// Potassium channel conductance
	double EK;			// Potassium reversal potential
	double gLeak;		// Leak conductance
	double EL;			// Leak reversal potential

	// Gating variable parameters
	double amV1, amV2, amV3;	// parameters for m-gate
	double tm0, epsm;
	double ahV1, ahV2, ahV3;	// parameters for h-gate
	double th0, epsh;
	double anV1, anV2, anV3;	// parameters for n-gate
	double tn0, epsn;

	// Initial conditions
	double V_init;				// Initial membrane potential
	double m_init;				// Initial m-gate value
	double h_init;				// Initial h-gate value
	double n_init;				// Initial n-gate value
	double init_conditions[4];	// Array to store initial conditions

	// ----------------------------------------------------------------------------------------------------
	// Results

	vector<double> X;
	vector<vector<double>> time_series;

	Eigen::MatrixXd t;
	Eigen::MatrixXd I_stim;
	Eigen::MatrixXd V_mem;
	Eigen::MatrixXd m_gate;
	Eigen::MatrixXd h_gate;
	Eigen::MatrixXd n_gate;

	// ----------------------------------------------------------------------------------------------------
	// Constructor & Destructor

	HodgkinHuxley(const HH_ParamStruct& configSelected);
	~HodgkinHuxley() = default;

	// ----------------------------------------------------------------------------------------------------
	// Load external stimulation files

	void loadStimulationCurrent();

	// ----------------------------------------------------------------------------------------------------
	// Gating Kinetics Functions

	double infSteadyState(double VV, double a1, double a2);
	double tauTimeConstant(double VV, double t0, double eps, double a1, double a3);
	double mm_inf(double VV);
	double mm_tau(double VV);
	double hh_inf(double VV);
	double hh_tau(double VV);
	double nn_inf(double VV);
	double nn_tau(double VV);

	// ----------------------------------------------------------------------------------------------------
	// Ionic Current Functions

	double I_Leak(double VV) const;
	double I_K(double VV, double nn) const;
	double I_NaT(double VV, double mm, double hh) const;

	// ----------------------------------------------------------------------------------------------------
	// Equations of Motion

	void dXdt(const vector<double>& X, vector<double>& dX, const int idx);

	// ----------------------------------------------------------------------------------------------------
	// Running the model

	void rkStepper(int i);
	void runModel();

	// ----------------------------------------------------------------------------------------------------
	// Export results the model

	void unpackTimeSeries();
	void exportResults();
};

#endif // HODGKIN_HUXLEY_H
