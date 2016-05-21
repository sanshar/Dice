#ifndef CIPSI_HEADER_H
#define CIPSI_HEADER_H
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <list>
#include <tuple>
#include <map>

using namespace std;
using namespace Eigen;
class Determinant;
class oneInt;
class twoInt;
class twoIntHeatBath;

namespace CIPSIbasics {
  struct schedule{
    double davidsonTol;
    double epsilon2;
    std::vector<double> epsilon1;    
    bool restart;
    bool fullrestart;
    double dE;
    double eps;
    string prefix;
    bool stochastic;
  };
  
  void writeVariationalResult(int iter, MatrixXd& ci, vector<Determinant>& Dets, map<Determinant,int>& SortedDets,
			      MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
			      double& E0, bool converged, string &file);
  void readVariationalResult(int& iter, MatrixXd& ci, vector<Determinant>& Dets, map<Determinant,int>& SortedDets,
			     MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
			     double& E0, bool& converged, string& file);

  double DoVariational(MatrixXd& ci, vector<Determinant>& Dets, schedule& schd,
		       twoInt& I2, twoIntHeatBath& I2HB, oneInt& I1, double& coreE);

  void DoPerturbativeStochastic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
				twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) ;
  void DoPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
				   twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) ;

  void updateConnections(vector<Determinant>& Dets, map<Determinant, int>& SortedDets, int norbs, oneInt& int1, twoInt& int2, double coreE, char* detChar, vector<vector<int> >& connections, vector<vector<double> >& Helements); 
  void getDeterminants(Determinant& d, double epsilon, double ci, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, double >& dets, std::vector<Determinant>& Psi0, bool additions=true);
  void getDeterminants(Determinant& d, double epsilon, double ci, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, pair<double,double> >& dets, std::vector<Determinant>& Psi0, int oneOrTwo);
  void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, double coreE, double E0, std::set<Determinant>& dets);
}

#endif
