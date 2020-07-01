#pragma once

#include "tensor.h"

double calcCoulombIntegral(int n1, double Ax, double Ay, double Az,
                           double expA, double normA,
                           double expG, double wtG,
                           int n2, double Bx, double By, double Bz,
                           double expB, double normB,
                           tensor& Int);
double calcOvlpMatrix(int LA, double Ax, double Ay, double Az, double expA,
                     int LB, double Bx, double By, double Bz, double expB,
                     tensor& S) ;


double calcCoulombIntegralPeriodic(int n1, double Ax, double Ay, double Az,
				   double expA, double normA,
				   int n2, double Bx, double By, double Bz,
				   double expB, double normB,
                                   double Lx, double Ly, double Lz,
				   tensor& Int, bool IncludeNorm=true);

double calcKineticIntegralPeriodic(int n1, double Ax, double Ay, double Az,
				   double expA, double normA,
				   int n2, double Bx, double By, double Bz,
				   double expB, double normB,
                                   double Lx, double Ly, double Lz,
				   tensor& Int, bool IncludeNorm=true);

void calcOverlapIntegralPeriodic(int LA, double Ax, double Ay, double Az,
                                 double expA, double normA,
                                 int LB, double Bx, double By, double Bz,
                                 double expB, double normB,
                                 double Lx, double Ly, double Lz,
                                 tensor& Int, bool IncludeNorm=true) ;

void calcShellIntegral_2c(char* Name, double* integrals, int sh1, int sh2, int* ao_loc,
                          int* atm, int natm, int* bas, int nbas,
                          double* env, double* Lattice);
void calcIntegral_2c(char* Name, double* integrals, int* shls, int* ao_loc,
                     int* atm, int natm, int* bas, int nbas,
                     double* env, double* Lattice);
