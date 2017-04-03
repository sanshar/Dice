#ifndef SHCI_HEADER_H
#define SHCI_HEADER_H
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

namespace SHCIbasics {
  void writeVariationalResult(int iter, vector<MatrixXx>& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
			      MatrixXx& diag, vector<vector<int> >& connections, vector<vector<size_t> >& orbDifference,
			      vector<vector<CItype> >& Helements, 
			      vector<double>& E0, bool converged, schedule& schd,   
			      std::map<HalfDet, std::vector<int> >& BetaN, 
			      std::map<HalfDet, std::vector<int> >& AlphaNm1);
  void readVariationalResult(int& iter, vector<MatrixXx>& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
			     MatrixXx& diag, vector<vector<int> >& connections, vector<vector<size_t> >& orbDifference, 
			     vector<vector<CItype> >& Helements, 
			     vector<double>& E0, bool& converged, schedule& schd,
			     std::map<HalfDet, std::vector<int> >& BetaN, 
			     std::map<HalfDet, std::vector<int> >& AlphaNm1);
  vector<double> DoVariational(vector<MatrixXx>& ci, vector<Determinant>& Dets, schedule& schd,
			       twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, oneInt& I1, double& coreE, int nelec,
			       bool DoRDM=false);
  
  void DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(vector<Determinant>& Dets, MatrixXx& ci
								 , double& E0, oneInt& I1, twoInt& I2, 
								 twoIntHeatBathSHM& I2HB,vector<int>& irrep,
								 schedule& schd, double coreE, 
								 int nelec, int root) ;
  void DoPerturbativeStochastic2SingleListDoubleEpsilon2OMPTogether(vector<Determinant>& Dets, MatrixXx& ci
								 , double& E0, oneInt& I1, twoInt& I2, 
								 twoIntHeatBathSHM& I2HB,vector<int>& irrep,
								 schedule& schd, double coreE, 
								 int nelec, int root) ;
  void DoPerturbativeStochastic2SingleListDoubleEpsilon2(vector<Determinant>& Dets, MatrixXx& ci
							 , double& E0, oneInt& I1, twoInt& I2, 
							 twoIntHeatBathSHM& I2HB,vector<int>& irrep,
							 schedule& schd, double coreE, 
							 int nelec, int root) ;
  
  void DoPerturbativeStochastic2SingleList(vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2, 
					   twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, int root) ;
  
  double DoPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2, 
				     twoIntHeatBathSHM& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, int root,
				     bool appendPsi1ToPsi0=false) ;
  
  void DoPerturbativeDeterministicOffdiagonal(vector<Determinant>& Dets, MatrixXx& ci1, double& E01, 
					      MatrixXx&ci2, double& E02, oneInt& I1, twoInt& I2,
					      twoIntHeatBathSHM& I2HB, vector<int>& irrep, 
					      schedule& schd, double coreE, int nelec, int root, 
					      CItype& EPT1, CItype& EPT2, CItype& EPT12,
					      std::vector<MatrixXx>& spinRDM);
    
  
  
  
  
  

}

#endif
