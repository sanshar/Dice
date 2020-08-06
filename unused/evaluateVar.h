#ifndef EvalVar_HEADER_H
#define EvalVar_HEADER_H
#include <vector>
#include <Eigen/Dense>
#include "Wfn.h"
#include <algorithm>
#include "integral.h"
#include "Determinants.h"
#include "Walker.h"
#include <boost/format.hpp>
#include <iostream>
#include "evaluateE.h"
#include "Davidson.h"
#include "Hmult.h"
#include <math.h>
#include "global.h"
#include "input.h"

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;
using namespace Eigen;


void getStochasticVarianceGradientContinuousTime(CPSSlater &w, double &E0, double &stddev,
                                                 int &nalpha, int &nbeta, int &norbs, oneInt &I1,
                                                 twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
                                                 Eigen::VectorXd &grad, double &var, double &rk,
						 int niter, double targetError);
#endif
