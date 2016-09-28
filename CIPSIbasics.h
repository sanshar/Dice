#ifndef CIPSI_HEADER_H
#define CIPSI_HEADER_H
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

namespace CIPSIbasics {
  enum AddTo {first, second, both};

  struct schedule{
 private:
  friend class boost::serialization::access;
  template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & davidsonTol \
	& epsilon2 \
	& SampleN \
	& epsilon1 \
	& onlyperturbative \
	& restart \
	& fullrestart \
	& dE \
	& eps \
	& prefix \
	& stochastic \
	& nblocks \
	& excitation \
	& nvirt \
	& singleList \
	& io;
    }

  public:
    double davidsonTol;
    double epsilon2;
    int SampleN;
    std::vector<double> epsilon1;    
    bool onlyperturbative;
    bool restart;
    bool fullrestart;
    double dE;
    double eps;
    string prefix;
    bool stochastic;
    int nblocks;
    int excitation;
    int nvirt;
    bool singleList;
    bool io;
  };
  
  void writeVariationalResult(int iter, MatrixXd& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
			      MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
			      double& E0, bool converged, schedule& schd,   
			      std::map<HalfDet, std::vector<int> >& BetaN, 
			      std::map<HalfDet, std::vector<int> >& AlphaNm1);

  void readVariationalResult(int& iter, MatrixXd& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
			     MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
			     double& E0, bool& converged, schedule& schd,
			     std::map<HalfDet, std::vector<int> >& BetaN, 
			     std::map<HalfDet, std::vector<int> >& AlphaNm1);


  double DoVariational(MatrixXd& ci, vector<Determinant>& Dets, schedule& schd,
		       twoInt& I2, twoIntHeatBath& I2HB, vector<int>& irrep, oneInt& I1, double& coreE);

  void DoBatchDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
			twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec);
  void DoPerturbativeStochastic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
				twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) ;
  void DoPerturbativeStochasticSingleList(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					  twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) ;
  void DoPerturbativeStochastic2(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
				twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) ;
  void DoPerturbativeStochastic2SingleList(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					   twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) ;
  void DoPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
				   twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) ;


  //used in batchdeterministic, doperturbativestochastic2, doperturbativestochastic
  //used in doperturbativestochastic2 when both the first and second list contain determinant d
  void getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, pair<double,double> >& dets, std::vector<Determinant>& Psi0, schedule& schd, int Nmc=0) ;
  void updateConnections(vector<Determinant>& Dets, map<Determinant,int>& SortedDets, int norbs, oneInt& int1, twoInt& int2, double coreE, char* detArray, vector<vector<int> >& connections, vector<vector<double> >& Helements) ;

}

#endif
