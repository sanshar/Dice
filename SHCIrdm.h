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
  void saveRDM(schedule& schd, MatrixXx& s2RDM, MatrixXx& twoRDM, int root);
  void loadRDM(schedule& schd, MatrixXx& s2RDM, MatrixXx& twoRDM, int root);

  void EvaluateRDM(vector<vector<int> >& connections, vector<Determinant>& Dets, MatrixXx& cibra,
		   MatrixXx& ciket, vector<vector<size_t> >& orbDifference, int nelec, schedule& schd, 
		   int root, MatrixXx& s2RDM, MatrixXx& twoRDM);

  void EvaluateOneRDM(vector<vector<int> >& connections, vector<Determinant>& Dets, MatrixXx& cibra,
		      MatrixXx& ciket, vector<vector<size_t> >& orbDifference, int nelec, schedule& schd, 
		      int root, MatrixXx& s1RDM);

  void UpdateRDMResponsePerturbativeDeterministic(vector<Determinant>& Dets, MatrixXx& ci, double& E0, 
						  oneInt& I1, twoInt& I2, schedule& schd, 
						  double coreE, int nelec, int norbs,
						  std::vector<SHCISortMpiUtils::StitchDEH>& uniqueDEH, int root,
						  double& Psi1Norm, MatrixXx& s2RDM, MatrixXx& twoRDM) ;

  void UpdateRDMPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXx& ci, double& E0, 
					  oneInt& I1, twoInt& I2, schedule& schd, 
					  double coreE, int nelec, int norbs,
					  std::vector<SHCISortMpiUtils::StitchDEH>& uniqueDEH,
					  int root,  MatrixXx& s2RDM, MatrixXx& twoRDM);
  
  void populateSpatialRDM(int& i, int& j, int& k, int& l, MatrixXx& s2RDM, 
			  CItype value, int& nSpatOrbs);


  double ComputeEnergyFromSpinRDM(int norbs, int nelec, oneInt& I1, twoInt& I2, 
				double coreE, MatrixXx& twoRDM);

  double ComputeEnergyFromSpatialRDM(int norbs, int nelec, oneInt& I1, twoInt& I2, 
				   double coreE, MatrixXx& twoRDM);

};

#endif
