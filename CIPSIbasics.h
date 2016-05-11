#ifndef CIPSI_HEADER_H
#define CIPSI_HEADER_H
#include <vector>

class Determinant;
class oneInt;
class twoInt;

void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::vector<Determinant>& dets);
#endif
