#ifndef DAVIDSON_HEADER_H
#define DAVIDSON_HEADER_H
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "global.h"

class Hmult2;
using namespace Eigen;
using namespace std;



void precondition(MatrixXx& r, MatrixXx& diag, double& e);
vector<double> davidson(Hmult2& H, vector<MatrixXx>& x0, MatrixXx& diag, int maxCopies, double tol, bool print);
double LinearSolver(Hmult2& H, MatrixXx& x0, MatrixXx& b, double tol, bool print);

#endif
