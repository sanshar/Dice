#include "interface.h"
#include "workArray.h"
#include "Integral3c.h"
#include "Integral2c.h"

void calcShellIntegralWrapper_3c(double* integrals, int sh1, int sh2, int sh3, int* ao_loc,
                              int* atm, int natm, int* bas, int nbas,
                              double* env, double* Lattice) {
  initWorkArray();
  calcShellIntegral(integrals, sh1, sh2, sh3, ao_loc, atm, natm, bas, nbas, env, Lattice);
}

void calcIntegralWrapper_3c(double* integrals, int* sh, int* ao_loc,
                            int* atm, int natm, int* bas, int nbas,
                            double* env, double* Lattice) {
  initWorkArray();
  calcIntegral_3c(integrals, sh, ao_loc, atm, natm, bas, nbas, env, Lattice);
}

void calcShellNuclearWrapper(double* integrals, int sh1, int sh2, int* ao_loc,
                              int* atm, int natm, int* bas, int nbas,
                              double* env, double* Lattice) {
  initWorkArray();
  calcShellNuclearIntegral(integrals, sh1, sh2, ao_loc, atm, natm, bas, nbas, env, Lattice);
}

void calcNuclearWrapper(double* integrals, int* sh, int* ao_loc,
                        int* atm, int natm, int* bas, int nbas,
                        double* env, double* Lattice) {
  initWorkArray();
  calcNuclearIntegral(integrals, sh, ao_loc, atm, natm, bas, nbas, env, Lattice);
}


void calcShellIntegralWrapper_2c(char* name, double* integrals, int sh1, int sh2, int* ao_loc,
                              int* atm, int natm, int* bas, int nbas,
                              double* env, double* Lattice) {
  initWorkArray();
  calcShellIntegral_2c(name, integrals, sh1, sh2, ao_loc, atm, natm, bas, nbas, env, Lattice);
}

void calcIntegralWrapper_2c(char* name, double* integrals, int* sh, int* ao_loc,
                            int* atm, int natm, int* bas, int nbas,
                            double* env, double* Lattice) {
  initWorkArray();
  calcIntegral_2c(name, integrals, sh, ao_loc, atm, natm, bas, nbas, env, Lattice);
}

