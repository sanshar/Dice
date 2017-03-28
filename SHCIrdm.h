#ifndef SHCI_RDM_H
#define SHCI_RDM_H
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <list>
#include <tuple>
#include <map>
#include <boost/serialization/serialization.hpp>

using namespace std;
using namespace Eigen;
class Determinant;
class HalfDet;
class oneInt;
class twoInt;
class twoIntHeatBath;
class twoIntHeatBathSHM;
class schedule;
namespace SHCISortMpiUtils{
  class StitchDEH;
};

namespace SHCIrdm {
  void EvaluateAndStoreRDM(vector<vector<int> >& connections, vector<Determinant>& Dets, MatrixXx& ci,
			   vector<vector<size_t> >& orbDifference, int nelec, schedule& schd, 
			   int root, bool update=false);

  void UpdateRDMPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXx& ci, double& E0, 
					  oneInt& I1, twoInt& I2, schedule& schd, 
					  double coreE, int nelec, int norbs,
					  std::vector<SHCISortMpiUtils::StitchDEH>& uniqueDEH, int root);
  
  void populateSpatialRDM(int& i, int& j, int& k, int& l, MatrixXx& s2RDM, 
			  CItype value, int& nSpatOrbs);

  void printRDM(int norbs, schedule& schd, int root, MatrixXx& twoRDM);

  void ComputeEnergyFromSpinRDM(int norbs, int nelec, oneInt& I1, twoInt& I2, 
				double coreE, MatrixXx& twoRDM);

  void ComputeEnergyFromSpatialRDM(int norbs, int nelec, oneInt& I1, twoInt& I2, 
				   double coreE, MatrixXx& twoRDM);

};

#endif
