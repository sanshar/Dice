#pragma once
#include "tensor.h"


double calc1DOvlp(int n1, double A, double expA,
                  int n2, double B, double expB,
                  tensor& S) ;
double calc1DOvlpPeriodic(int n1, double A, double expA,
                          int n2, double B, double expB,
                          double t, int AdditionalDeriv, 
                          double L, 
                          tensor& S, double* workArray,
                          tensor& powPiOverL, tensor& pow1OverExpA,
                          tensor& pow1OverExpB) ;

//actually it is \sqrt(pi/t) Theta3[z, exp(-pi*pi/t)]
double JacobiTheta(double z, double t, bool includeZero = true);
void JacobiThetaDerivative(double z, double t, double* workArray, int nderiv, bool includeZero=true) ;

double nChoosek( size_t n, size_t k );
double doubleFact(size_t n) ;
