#pragma once

#include "tensor.h"

class Coulomb;

void generateCoefficientMatrix(int LA, int LB, double expA, double expB,
                               double ABx, double p, tensor& Coeff_3d);

void calcCoulombIntegralPeriodic_noTranslations(
    int n1, double Ax, double Ay, double Az,
    double expA, double normA,
    int n2, double Bx, double By, double Bz,
    double expB, double normB,
    int n3, double Cx, double Cy, double Dz,
    double expC, double normC,
    double Lx, double Ly, double Lz,
    tensor& Int, Coulomb& coulomb,
    bool normalize = true);

void calcCoulombIntegralPeriodic_BTranslations(
    int n1, double Ax, double Ay, double Az,
    double expA, double normA,
    int n2, double Bx, double By, double Bz,
    double expB, double normB,
    int n3, double Cx, double Cy, double Dz,
    double expC, double normC,
    double Lx, double Ly, double Lz,
    tensor& Int, Coulomb& coulomb,
    bool normalize = true);

void calcShellIntegral(double* integrals, int sh1, int sh2, int sh3, int* ao_loc,
                       int* atm, int natm, int* bas, int nbas,
                       double* env, double* Lattice);

void calcIntegral_3c(double* integrals, int* sh1, int* ao_loc,
                       int* atm, int natm, int* bas, int nbas,
                       double* env, double* Lattice);

void calcShellNuclearIntegral(double* integrals, int sh1, int sh2, int* ao_loc,
                              int* atm, int natm, int* bas, int nbas,
                              double* env, double* Lattice);
void calcNuclearIntegral(double* integrals, int* shls, int* ao_loc,
                         int* atm, int natm, int* bas, int nbas,
                         double* env, double* Lattice);
