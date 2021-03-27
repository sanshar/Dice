#pragma once
#include <vector>
#include "CxMemoryStack.h"

class BasisShell;
class BasisSet;

struct LatticeSum {

  double RVolume, KVolume, Rscreen, Kscreen;
  std::vector<double> RLattice, KLattice;
  
  std::vector<double> Rcoord;
  std::vector<double> Rdist;

  std::vector<double> Kcoord;
  std::vector<int> Kcoordindex;
  std::vector<double> Kdist;
  std::vector<std::vector<long>> KSumIdx;
  std::vector<std::vector<size_t>> ROrderedIdx;
  std::vector<double> KSumVal;
  std::vector<double> atomCenters;
  std::vector<std::vector<double>> CosKval3c;
  std::vector<std::vector<double>> SinKval3c;
  
  double Eta2RhoOvlp, Eta2RhoCoul;
  LatticeSum(double* Lattice, int nr, int nk, ct::FMemoryStack& Mem,
             BasisSet& basis, double _Eta2Rho=100.0, double _Eta2RhoCoul = 8.0,
	     double Rscreen=1.e-9, double Kscreen=1e-12, bool make2cIntermediates=true,
             bool make3cIntermediates=true);
  void getRelativeCoords(BasisShell *pA, BasisShell *pC,
                         double& Tx, double& Ty, double& Tz);
  void printLattice();
  void getIncreasingIndex(std::vector<size_t>& idx, double Tx, double Ty, double Tz, ct::FMemoryStack& Mem) ;
  void getIncreasingIndex(size_t *&inx, double Tx, double Ty, double Tz, ct::FMemoryStack& Mem) ;
  void makeKsum2c(BasisSet& basis);
  void makeKsum3c(BasisSet& basis);
  int indexCenter(BasisShell& bas);
  void OrderedLatticeSumForEachAtomPair(BasisSet& basis, ct::FMemoryStack& Mem);
};
