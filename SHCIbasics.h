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

  
  void regenerateH(std::vector<Determinant>& Dets, 
		   std::vector<std::vector<int> >&connections,
		   std::vector<std::vector<CItype> >& Helements,
		   oneInt& I1,
		   twoInt& I2,
		   double& coreE);

  void PopulateHelperLists(std::map<HalfDet, std::vector<int> >& BetaN,
			   std::map<HalfDet, std::vector<int> >& AlphaNm1,
			   std::vector<Determinant>& Dets,
			   int StartIndex);
  
  
  void MakeHfromHelpers(std::map<HalfDet, std::vector<int> >& BetaN,
			std::map<HalfDet, std::vector<int> >& AlphaNm1,
			std::vector<Determinant>& Dets,
			int StartIndex,
			std::vector<std::vector<int> >&connections,
			std::vector<std::vector<CItype> >& Helements,
			int Norbs,
			oneInt& I1,
			twoInt& I2,
			double& coreE,
			std::vector<std::vector<size_t> >& orbDifference,
			bool DoRDM=false) ;
  
  
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
					      CItype& EPT1, CItype& EPT2, CItype& EPT12);
    
  
  
  void updateSOCconnections(vector<Determinant>& Dets, int prevSize, vector<vector<int> >& connections, vector<vector<CItype> >& Helements, int norbs, oneInt& int1, int nelec, bool includeSz=true);
  
  
  
  
  void DoBatchDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
			twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec);
  void DoPerturbativeStochastic(vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2, 
				twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) ;
  void DoPerturbativeStochasticSingleList(vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2, 
					  twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) ;
  void DoPerturbativeStochastic2(vector<Determinant>& Dets, MatrixXx& ci, double& E0, oneInt& I1, twoInt& I2, 
				twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) ;


  //used in batchdeterministic, doperturbativestochastic2, doperturbativestochastic
  //used in doperturbativestochastic2 when both the first and second list contain determinant d
  void getDeterminants(Determinant& d, double epsilon, CItype ci1, CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, pair<double,double> >& dets, std::vector<Determinant>& Psi0, schedule& schd, int Nmc=0) ;
  void getDeterminants(Determinant& d, double epsilon, CItype ci1, CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, tuple<double,double,double> >& Psi1, std::vector<Determinant>& Psi0, schedule& schd, int Nmc, int nelec);
  void getDeterminants2Epsilon(Determinant& d, double epsilon, double epsilonLarge, CItype ci1, CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, tuple<double, double,double,double,double> >& Psi1, std::vector<Determinant>& Psi0, schedule& schd, int Nmc, int nelec);
  void updateConnections(vector<Determinant>& Dets, map<Determinant,int>& SortedDets, int norbs, oneInt& int1, twoInt& int2, double coreE, char* detArray, vector<vector<int> >& connections, vector<vector<double> >& Helements) ;


  
  void getDeterminants2Epsilon(Determinant& d, double epsilon, double epsilonLarge, CItype ci1, CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<double>& numerator1A, vector<double>& numerator2A, vector<bool>& present, std::vector<double>& energy, schedule& schd, int Nmc, int nelec);

}

#endif
