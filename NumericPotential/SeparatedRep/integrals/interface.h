#pragma once
#include <vector>

//#### LIBCINT VARIABLES ######
/*
extern int *shls, *ao_loc, *atm, *bas;
extern int natm, nbas, ncalls;
extern double *env, *dm, *centroid, coordScale, *lattice;
extern std::vector<unsigned char> non0tab;
extern int BLKSIZE;
extern std::vector<double> aovals, intermediate, grid_fformat;
*/

extern "C" {
  void getDensityValuesAtGrid(const double* coords, int n, double* out); 
  void getAoValuesAtGrid(const double* grid_cformat, int ngrid, double* aovals);
  void getDensityFittedPQIntegrals();

  void initPeriodic(int* pshls, int *pao_loc, int *patm, int pnatm,
                    int *pbas, int pnbas, double *penv,
                    double* lattice);

  //3 center integral interface
  void calcShellIntegralWrapper_3c(double* integrals, int sh1, int sh2, int sh3, int* ao_loc,
                                int* atm, int natm, int* bas, int nbas,
                                double* env, double* Lattice) ;
  void calcIntegralWrapper_3c(double* integrals, int *shls, int* ao_loc,
                              int* atm, int natm, int* bas, int nbas,
                              double* env, double* Lattice) ;

  //nuclear integral for shell
  void calcShellNuclearWrapper(double* integrals, int sh1, int sh2, int* ao_loc,
                               int* atm, int natm, int* bas, int nbas,
                               double* env, double* Lattice) ;
  void calcNuclearWrapper(double* integrals, int* sh, int* ao_loc,
                          int* atm, int natm, int* bas, int nbas,
                          double* env, double* Lattice) ;

  //2 center integral interface
  void calcShellIntegralWrapper_2c(char* name, double* integrals, int sh1,
                                   int sh2, int* ao_loc,
                                   int* atm, int natm, int* bas, int nbas,
                                   double* env, double* Lattice) ;
  void calcIntegralWrapper_2c(char* name, double* integrals, int *shls, int* ao_loc,
                              int* atm, int natm, int* bas, int nbas,
                              double* env, double* Lattice) ;
  
}
