#ifndef LORENZ_OSCILLATOR_H
#define LORENZ_OSCILLATOR_H

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


class LorenzOscillator {
private:
public:
    // ----------------------------------------------------------------------------------------------------
    // Parameters

    // Runge-Kutta Cash-Karp 5(4) for integration
    runge_kutta_cash_karp54<vector<double>> rkck;


    // Export filepath
    string export_filepath;

    // Model constants
    double sigma;
    double rho;
    double beta;

    // Model parameters
    double T;            // Model duration in ms
    double dt;           // Time step in ms
    int num_steps;

    // Initial conditions
    double x_init;
    double y_init;
    double z_init;
    double init_conditions[3];    // Array to store initial conditions

    // ----------------------------------------------------------------------------------------------------
    // Results

    vector<double> X;
    vector<vector<double>> time_series;

    Eigen::MatrixXd t;
    Eigen::MatrixXd x_vals;
    Eigen::MatrixXd y_vals;
    Eigen::MatrixXd z_vals;

    // ----------------------------------------------------------------------------------------------------
    // Constructor & Destructor

    LorenzOscillator(const Lorenz_ParamStruct& configSelected);
    ~LorenzOscillator() = default;

    // ----------------------------------------------------------------------------------------------------
    // Equations of Motion

    void dXdt(const vector<double>& X, vector<double>& dX, const double t);

    // ----------------------------------------------------------------------------------------------------
    // Running the model

    void rkStepper(int i);
    void runModel();

    // ----------------------------------------------------------------------------------------------------
    // Helper functions

    void unpackTimeSeries();

    // ----------------------------------------------------------------------------------------------------
    // Export results from the model

    void exportResults();
};

#endif // LORENZ_OSCILLATOR_H