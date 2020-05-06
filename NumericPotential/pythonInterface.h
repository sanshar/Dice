#pragma once
#include <vector>
#include "fittedFunctions.h"

using namespace std;
using namespace Eigen;
using namespace boost;

extern "C" {
  //get density at grid points
  void GTOval_cart(int ngrids, int *shls_slice, int *ao_loc,
                   double *ao, double *coord, char *non0table,
                   int *atm, int natm, int *bas, int nbas, double *env);
  
  void getValuesAtGrid(const double* coords, int n, double* out); 


  //FMM functions
  void getFockFMM(double* pdm, double* fock, double tol);
  void initFMMGridAndTree(double* pdm, double* centroid, double scale, double tol);
  void getPotentialFMM(int n, double* coords, double* potential, double* pdm,
                       double* centroid, double scale, double tol);


  //Becke grid functions
  void VXCgen_grid(double* pbecke, double* coords, double* atmcoord,
                   double* p_radii_table, int natm, int ngrid);
  

  void init(int* pshls, int *pao_loc, int *patm, int pnatm,
            int *pbas, int pnbas, double *penv);

  void getPotentialBecke(int ngird, double* grid, double* potential,
                         int lmax, int* nrad, double* rm, double* pdm);
  void getPotentialBeckeOfAtom(int ia, int ngird, double* grid, double* potential);
}

//miscellaneous functions
void initAtomGrids(int natom, double* atomCoords);
void initAngularData(int lmax, int lebdevOrder, double* angGrid, double* angWts);

void initDensityRadialGrid(int ia, int rmax, double rm, double* radialGrid, double* radialWts);
void fitDensity(int ia, int rindex, double* density);
//void getDensityOnLebdevGridAtGivenR(int rindex, double* density);
void solvePoissonsEquation(int ia);
void fitSplinePotential(int ia);


