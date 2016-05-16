#ifndef CIPSI_HEADER_H
#define CIPSI_HEADER_H
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <list>
#include <tuple>
#include <map>




class Determinant;
class oneInt;
class twoInt;
class twoIntHeatBath;

void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::list<Determinant>& dets);
void getDeterminants(Determinant& d, double epsilon, double ci, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, double >& dets, std::vector<Determinant>& Psi0, bool additions=true);
void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, double coreE, double E0, std::set<Determinant>& dets);
void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::vector<Determinant>& dets);
void getDeterminants(std::vector<Determinant>& detsIn, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::set<Determinant>& dets,Eigen::MatrixXd& ci);
#endif
