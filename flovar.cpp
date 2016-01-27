#include <math.h>
#include "globals.h"
#include "flovar.h"

// Gas Constants
const double gam = 1.4;
const double R = 1716.0;
const double Cv = R / (gam - 1);

// Inlet
const double Min = 2.2;
const double Ttin = 531.2;
const double ptin = 2117;
// Outlet
const double pexit = 0.72 * ptin;
// Constant
const double a2 = 2 * gam * Cv * Ttin * ((gam - 1) / (gam + 1)); // used in isentropic nozzle

// Convergence Settings
const double CFL = 0.5;
const double conv = 1e-13;
const int maxIt = 50000;
const int printIt = 100;
const int printConv = 0; // 0 to hide real - time convergence
const int printW = 1;

// Stepping Scheme
// 0   -   Euler Explicit
// 1   -   Runge - Kutta 4th order
// 2   -   Jameson's Runge-Kutta 4th order
const int StepScheme = 2;

// Flux Scheme
// 0   -   Steger Warming (SW)
// 1   -   Scalar Dissipation
const int FluxScheme = 0;
