#include "data_HodgkinHuxley.h"

// ----------------------------------------------------------------------------------------------------
// Constructor & Destructor

HodgkinHuxley::HodgkinHuxley(const HH_ParamStruct& configSelected) {

    cout << "\nData Generation: Hodgkin Huxley ----------------" << endl;

    // Unpack the config struct.

    export_filepath = configSelected.export_filepath;

    stimFilePaths = configSelected.stimFilePaths;
    stimScale = configSelected.stimScale;
    samplingInterval = configSelected.samplingInterval;

    TEMP_C = configSelected.TEMP_C;

    T = configSelected.T;
    dt = configSelected.dt;
    num_steps = static_cast<int>(T / dt);

    soma_len = configSelected.soma_len;
    soma_diam = configSelected.soma_diam;
    Cm = configSelected.Cm;

    gNaT = configSelected.gNaT;
    ENa = configSelected.ENa;
    gK = configSelected.gK;
    EK = configSelected.EK;
    gLeak = configSelected.gLeak;
    EL = configSelected.EL;

    amV1 = configSelected.amV1;
    amV2 = configSelected.amV2;
    amV3 = configSelected.amV3;
    tm0 = configSelected.tm0;
    epsm = configSelected.epsm;

    ahV1 = configSelected.ahV1;
    ahV2 = configSelected.ahV2;
    ahV3 = configSelected.ahV3;
    th0 = configSelected.th0;
    epsh = configSelected.epsh;

    anV1 = configSelected.anV1;
    anV2 = configSelected.anV2;
    anV3 = configSelected.anV3;
    tn0 = configSelected.tn0;
    epsn = configSelected.epsn;

    V_init = configSelected.V_init;
    m_init = configSelected.m_init;
    h_init = configSelected.h_init;
    n_init = configSelected.n_init;

    init_conditions[0] = V_init;
    init_conditions[1] = m_init;
    init_conditions[2] = h_init;
    init_conditions[3] = n_init;

    X.resize(4);
    copy(init_conditions, init_conditions + 4, X.begin());

    // Time series for all variables with one extra to hold t
    time_series.resize(num_steps, vector<double>(6));

    loadStimulationCurrent();

};

// ----------------------------------------------------------------------------------------------------
// Load external stimulation files

void HodgkinHuxley::loadStimulationCurrent() {

    cout << "Loading HH stimulation current from files" << endl;

    for (string filePath : stimFilePaths) {
        ifstream file(filePath);

        // Check if opened successfully.
        if (!file.is_open()) {
            cerr << "Error: Could not open file: " << filePath << endl;
            exit(0);
        }

        string line;
        int counter = 0;

        while (getline(file, line)) {

            // Only nth lines are chosen.
            if (counter % samplingInterval == 0) {

                // Scale and add to array. (String_to_double is stod)
                stim.push_back(stimScale * stod(line));
            }
            counter++;

        }
    }

    cout << "Loading complete" << endl;
}

// ----------------------------------------------------------------------------------------------------
// Gating Kinetics Functions

double HodgkinHuxley::infSteadyState(double VV, double a1, double a2) {
	return 0.5 * (1 + tanh((VV - a1) / a2));
}
double HodgkinHuxley::tauTimeConstant(double VV, double t0, double eps, double a1, double a3) {
	return (t0 + eps * (1 - tanh((VV - a1) / a3) * tanh((VV - a1) / a3))) / pow(3.0, (TEMP_C - 23.5) / 10);
}

// mm (sodium activation)
double HodgkinHuxley::mm_inf(double VV) { return infSteadyState(VV, amV1, amV2); }
double HodgkinHuxley::mm_tau(double VV) { return tauTimeConstant(VV, tm0, epsm, amV1, amV3); }

// hh (sodium inactivation)
double HodgkinHuxley::hh_inf(double VV) { return infSteadyState(VV, ahV1, ahV2); }
double HodgkinHuxley::hh_tau(double VV) { return tauTimeConstant(VV, th0, epsh, ahV1, ahV3); }

// nn (potassium activation)
double HodgkinHuxley::nn_inf(double VV) { return infSteadyState(VV, anV1, anV2); }
double HodgkinHuxley::nn_tau(double VV) { return tauTimeConstant(VV, tn0, epsn, anV1, anV3); }

// ----------------------------------------------------------------------------------------------------
// Ionic Current Functions

double HodgkinHuxley::I_Leak(double VV) const {
	return gLeak * (VV - EL);
}
double HodgkinHuxley::I_K(double VV, double nn) const {
	return gK * pow(nn, 4) * (VV - EK);
}
double HodgkinHuxley::I_NaT(double VV, double mm, double hh) const {
	return gNaT * pow(mm, 3) * hh * (VV - ENa);
}

// ----------------------------------------------------------------------------------------------------
// Equations of Motion

void HodgkinHuxley::dXdt(const vector<double>& X, vector<double>& dX, const int idx) {

	// Unpack for context.
	double VV = X[0];
	double mm = X[1];
	double hh = X[2];
	double nn = X[3];

	double soma_area = soma_len * soma_diam * PI;

	// Calculate derivatives
	dX[0] = (-(I_NaT(VV, mm, hh) + I_K(VV, nn) + I_Leak(VV)) + stim[idx] / soma_area) / Cm; // dVV/dt
	dX[1] = (mm_inf(VV) - mm) / mm_tau(VV);		// dm/dt
	dX[2] = (hh_inf(VV) - hh) / hh_tau(VV);		// dh/dt
	dX[3] = (nn_inf(VV) - nn) / nn_tau(VV);		// dn/dt
}

// ----------------------------------------------------------------------------------------------------
// Running the model

void HodgkinHuxley::rkStepper(int i) {

	double t = i * dt;
	time_series[i][0] = t;
	time_series[i][1] = stim[i];

	// Integrate with RK 
	rkck.do_step(
		[this](const vector<double>& X, vector<double>& dX, const int idx)
		{
			this->dXdt(X, dX, idx);
		},
		X, i, dt
	);

	// Store updated variables
	for (int j = 0; j < 4; ++j) {
		time_series[i][j + 2] = X[j];
	}
}

void HodgkinHuxley::runModel() {

	cout << "Starting HH Model" << endl;

	int num_steps = static_cast<int>(T / dt);

	// time series for all variables w One extra to hold t.
	time_series.resize(num_steps, vector<double>(6));


	for (int i = 0; i < num_steps; ++i) {

		progressBar(i, num_steps);
		rkStepper(i);

	}

	unpackTimeSeries();
	cout << "Progress: 100%" << endl;
}

// ----------------------------------------------------------------------------------------------------
// Export Results

void HodgkinHuxley::unpackTimeSeries() {

	int num_steps = time_series.size();

	t.resize(1, num_steps);
	I_stim.resize(1, num_steps);
	V_mem.resize(1, num_steps);
	m_gate.resize(1, num_steps);
	h_gate.resize(1, num_steps);
	n_gate.resize(1, num_steps);

	for (int i = 0; i < num_steps; ++i) {
		t(0, i) = time_series[i][0];			// t
		I_stim(0, i) = time_series[i][1];		// I_stim
		V_mem(0, i) = time_series[i][2];		// V
		m_gate(0, i) = time_series[i][3];		// m
		h_gate(0, i) = time_series[i][4];		// h
		n_gate(0, i) = time_series[i][5];		// n
	}

}

void HodgkinHuxley::exportResults() {

	string header = "t_step, t, I_stim, V_mem, m, h, n";

	writeVectorofVectorToCSV(export_filepath, time_series, header);
}