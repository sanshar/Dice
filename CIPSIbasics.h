#ifndef CIPSI_HEADER_H
#define CIPSI_HEADER_H
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <list>

class Determinant;
class oneInt;
class twoInt;

void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::list<Determinant>& dets);
void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::set<Determinant>& dets);
void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::vector<Determinant>& dets);
void getDeterminants(std::vector<Determinant>& detsIn, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::set<Determinant>& dets,Eigen::MatrixXd& ci);
#endif
