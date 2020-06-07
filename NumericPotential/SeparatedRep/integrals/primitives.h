#pragma once
#include <Eigen/Dense>

using namespace Eigen;


double calc1DOvlp(int n1, double A, double expA,
                  int n2, double B, double expB,
                  MatrixXd& S) ;
double calc1DOvlpPeriodic(int n1, double A, double expA,
                          int n2, double B, double expB,
                          MatrixXd& S) ;

//actually it is \sqrt(pi/t) Theta3[z, exp(-pi*pi/t)]
double JacobiTheta(double z, double t, bool includeZero = true);

double nChoosek( size_t n, size_t k );
double doubleFact(size_t n) ;
