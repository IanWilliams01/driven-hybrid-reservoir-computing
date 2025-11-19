#include "data_LorenzOscillator.h"

// ----------------------------------------------------------------------------------------------------
// Constructor & Destructor

LorenzOscillator::LorenzOscillator(const Lorenz_ParamStruct& configSelected) {

    cout << "\nData Generation: Lorenz Oscillator -------------" << endl;

    // Unpack the config struct.
    export_filepath = configSelected.export_filepath;

    sigma = configSelected.sigma;
    rho = configSelected.rho;
    beta = configSelected.beta;

    T = configSelected.T;
    dt = configSelected.dt;
    num_steps = static_cast<int>(T / dt);

    x_init = configSelected.x_init;
    y_init = configSelected.y_init;
    z_init = configSelected.z_init;

    init_conditions[0] = x_init;
    init_conditions[1] = y_init;
    init_conditions[2] = z_init;

    X.resize(3);
    copy(init_conditions, init_conditions + 3, X.begin());

    // Time series for all variables with one extra to hold t
    time_series.resize(num_steps, vector<double>(4));
}

// ----------------------------------------------------------------------------------------------------
// Equations of Motion

void LorenzOscillator::dXdt(const vector<double>& X, vector<double>& dX, const double t) {

	// Unpack for context.
	double x = X[0];
	double y = X[1];
	double z = X[2];

	// Calculate derivatives
	dX[0] = sigma * (y - x);			// dx/dt
	dX[1] = (x * (rho - z)) - y;		// dy/dt
	dX[2] = (x * y) - (beta * z);		// dz/dt

}


// ----------------------------------------------------------------------------------------------------
// Running the model

void LorenzOscillator::rkStepper(int i) {

	double t = i * dt;
	time_series[i][0] = t;

	// Integrate with RK
	rkck.do_step(
		[this](const vector<double>& X, vector<double>& dX, const double t) {
			this->dXdt(X, dX, t);
		},
		X, t, dt
	);

	// Store updated variables (+1 as t stored too)
	for (int j = 0; j < 3; ++j) {
		time_series[i][j + 1] = X[j];
	}
}

void LorenzOscillator::runModel() {

	cout << "Starting Lorentz Model" << endl;;

	for (int i = 0; i < num_steps; ++i) {

		progressBar(i, num_steps);
		rkStepper(i);

	}

	unpackTimeSeries();
	cout << "Progress: 100%" << endl;
};

void LorenzOscillator::unpackTimeSeries() {

	int num_steps = time_series.size();

	t.resize(1, num_steps);
	x_vals.resize(1, num_steps);
	y_vals.resize(1, num_steps);
	z_vals.resize(1, num_steps);


	for (int i = 0; i < num_steps; ++i) {
		t(0, i) = time_series[i][0];			// t
		x_vals(0, i) = time_series[i][1];		// x
		y_vals(0, i) = time_series[i][2];		// y
		z_vals(0, i) = time_series[i][3];		// z
	}

}

// ----------------------------------------------------------------------------------------------------
// Export Results 

void LorenzOscillator::exportResults() {

	string header = "t_step, t, x, y, z";

	writeVectorofVectorToCSV(export_filepath, time_series, header);
}
