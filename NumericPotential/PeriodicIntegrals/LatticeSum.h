#pragma once
#include <vector>
#include "CxMemoryStack.h"

class BasisShell;
class BasisSet;

struct LatticeSum {

  double RVolume, KVolume, screen;
  std::vector<double> RLattice, KLattice;
  
  std::vector<double> Rcoord;
  std::vector<double> Rdist;

  std::vector<double> Kcoord;
  std::vector<double> Kdist;
  std::vector<std::vector<long>> KSumIdx;
  std::vector<double> KSumVal;
  std::vector<double> atomCenters;
  
  double Eta2RhoOvlp, Eta2RhoCoul;
  LatticeSum(double* Lattice, int nr, int nk, double _Eta2Rho=100.0, double _Eta2RhoCoul = 8.0, double screen=1.e-12);
  void getRelativeCoords(BasisShell *pA, BasisShell *pC,
                         double& Tx, double& Ty, double& Tz);
  void printLattice();
  void getIncreasingIndex(size_t *&inx, double Tx, double Ty, double Tz, ct::FMemoryStack& Mem) ;
  void makeKsum(BasisSet& basis);
  int indexCenter(BasisShell& bas);
};
