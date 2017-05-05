#include "Determinants.h"
#include "SHCIgetdeterminants.h"
#include "input.h"
#include "integral.h"
#include <vector>
#include "math.h"
#include "Hmult.h"
#include <tuple>
#include <map>
#include "Davidson.h"
#include "boost/format.hpp"
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace boost;



void SHCIgetdeterminants::getDeterminantsDeterministicPT(Determinant& d, double epsilon, CItype ci1, CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<CItype>& numerator, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  vector<int> closed(nelec,0);
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  //d.getRepArray(detArray);

  double Energyd = d.Energy(int1, int2, coreE);

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    //CItype integral = d.Hij_1Excite(closed[i],open[a],int1,int2);
    if (closed[i]%2 != open[a]%2 || irreps[closed[i]/2] != irreps[open[a]/2]) continue;
    CItype integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);

    if (abs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);

      numerator.push_back(integral*ci1);
      energy.push_back(E);
    }
  }

  if (abs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X*(X+1)/2+Y;
    size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
    double* integrals = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
    int* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;


    for (size_t index=start; index<end; index++) {
      if (abs(integrals[index]) <epsilon) break;
      int a = 2* orbIndices[2*index] + closed[i]%2, b= 2*orbIndices[2*index+1]+closed[j]%2;

      if (!(d.getocc(a) || d.getocc(b))) {
	dets.push_back(d);
	Determinant& di = *dets.rbegin();
	di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);

	double sgn = 1.0;
	di.parity(a, b, closed[i], closed[j], sgn);
	numerator.push_back(integrals[index]*sgn*ci1);

	double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);
	energy.push_back(E);
      }
    }
  }
  return;
}

void SHCIgetdeterminants::getDeterminantsDeterministicPTKeepRefDets(Determinant det, int det_ind, double epsilon, CItype ci, oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<CItype>& numerator, std::vector<double>& energy, std::vector<int>& var_indices, std::vector<size_t>& orbDifference, schedule& schd, int nelec) {
  // Similar to above subroutine, but also keeps track of the reference dets each connected det came from
  // AAH, 30 Jan 2017

  int norbs = det.norbs;
  //char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0);
  vector<int> open(norbs-nelec,0);
  det.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  size_t orbDiff;
  //d.getRepArray(detArray);

  double Energyd = det.Energy(int1, int2, coreE);
  std::vector<int> var_indices_vec;
  std::vector<size_t> orbDiff_vec;

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    //if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;

    CItype integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);


    if (fabs(integral) > epsilon ) {
      dets.push_back(det); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);

      numerator.push_back(integral*ci);
      energy.push_back(E);

      var_indices.push_back(det_ind);

      size_t A = open[a], N= norbs, I = closed[i];
      orbDiff = A*N+I; // a = creation, i = annihilation
      orbDifference.push_back(orbDiff);

    }
  }

  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X*(X+1)/2+Y;
    size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
    double* integrals = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
    int* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;


    for (size_t index=start; index<end; index++) {
      if (fabs(integrals[index]) <epsilon) break;
      int a = 2* orbIndices[2*index] + closed[i]%2, b= 2*orbIndices[2*index+1]+closed[j]%2;

      if (!(det.getocc(a) || det.getocc(b))) {
	dets.push_back(det);
	Determinant& di = *dets.rbegin();
	di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);

	double sgn = 1.0;
	di.parity(a, b, closed[i], closed[j], sgn);
	numerator.push_back(integrals[index]*sgn*ci);

	double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);
	energy.push_back(E);

	var_indices.push_back(det_ind);

	size_t A = a, B = b, N= norbs, I = closed[i], J = closed[j];
	orbDiff = A*N*N*N+I*N*N+B*N+J;  //i>j and a>b??
	orbDifference.push_back(orbDiff);

      } // if a and b unoccupied
    } // iterate over heatbath integrals
  } // i and j
  return;
}


void SHCIgetdeterminants::getDeterminantsDeterministicPTWithSOC(Determinant det, int det_ind, double epsilon1, CItype ci1, double epsilon2, CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE, std::vector<Determinant>& dets, std::vector<CItype>& numerator1, std::vector<CItype>& numerator2, std::vector<double>& energy, schedule& schd, int nelec) {

  int norbs = det.norbs;
  vector<int> closed(nelec,0);
  vector<int> open(norbs-nelec,0);
  det.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  size_t orbDiff;

  double Energyd = det.Energy(int1, int2, coreE);
  std::vector<int> var_indices_vec;
  std::vector<size_t> orbDiff_vec;

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;

    CItype integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);
    if (closed[i]%2 != open[a]%2) {
      double sgn = 1.0;
      det.parity(min(open[a],closed[i]), max(open[a],closed[i]),sgn);
      integral = int1(open[a], closed[i])*sgn;
    }
    //integral = int1(closed[i], open[a]);

    if (fabs(integral) > epsilon1 || fabs(integral) > epsilon2 ) {
      dets.push_back(det); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);
      //double E = Energyd - int1(closed[i], closed[i]) + int1(open[a],open[a]);
      if (closed[i]%2 != open[a]%2)
	E = di.Energy(int1, int2, coreE);
      energy.push_back(E);

      if(fabs(integral) >epsilon1)
	numerator1.push_back(integral*ci1);
      else
	numerator1.push_back(0.0);

      if(fabs(integral) > epsilon2)
	numerator2.push_back(integral*ci2);
      else
	numerator2.push_back(0.0);
    }
  }

  if (fabs(int2.maxEntry) <epsilon1 && fabs(int2.maxEntry) < epsilon2) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X*(X+1)/2+Y;
    size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
    double* integrals = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
    int* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;


    for (size_t index=start; index<end; index++) {
      if (fabs(integrals[index]) <epsilon1 && fabs(integrals[index]) <epsilon2) break;
      int a = 2* orbIndices[2*index] + closed[i]%2, b= 2*orbIndices[2*index+1]+closed[j]%2;

      if (!(det.getocc(a) || det.getocc(b))) {
	dets.push_back(det);
	Determinant& di = *dets.rbegin();
	di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);

	double sgn = 1.0;
	di.parity(a, b, closed[i], closed[j], sgn);
	double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);
	energy.push_back(E);

	//cout << di<<" gen2 "<<E<<endl;

	if(fabs(integrals[index]) >epsilon1)
	  numerator1.push_back(integrals[index]*sgn*ci1);
	else
	  numerator1.push_back(0.0);
	
	if(fabs(integrals[index]) > epsilon2)
	  numerator2.push_back(integrals[index]*sgn*ci2);
	else
	  numerator2.push_back(0.0);


      } // if a and b unoccupied
    } // iterate over heatbath integrals
  } // i and j
  return;
}



void SHCIgetdeterminants::getDeterminantsVariational(Determinant& d, double epsilon, CItype ci1, CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, schedule& schd, int Nmc, int nelec) {

  //Make the int represenation of open and closed orbitals of determinant
  //this helps to speed up the energy calculation
  int norbs = d.norbs;
  vector<int> closed(nelec,0);
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;


  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;

    //if we are doing SOC calculation then breaking spin and point group symmetry is allowed
#ifndef Complex 
    if (closed[i]%2 != open[a]%2 || irreps[closed[i]/2] != irreps[open[a]/2]) continue;
#endif

    CItype integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);

    if (closed[i]%2 != open[a]%2) {
      double sgn = 1.0;
      d.parity(min(open[a],closed[i]), max(open[a],closed[i]),sgn);
      integral = int1(open[a], closed[i])*sgn; 
    }

    if (closed[i]%2 != open[a]%2 || irreps[closed[i]/2] != irreps[open[a]/2])
      integral *= schd.socmultiplier; //make it 100 times so SOC gets preference


    if (abs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);
    }
  }

  if (abs(int2.maxEntry) <epsilon) return;


  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X*(X+1)/2+Y;
    size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
    double* integrals = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
    int* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;


    for (size_t index=start; index<end; index++) {
      if (abs(integrals[index]) <epsilon) break;
      int a = 2* orbIndices[2*index] + closed[i]%2, b= 2*orbIndices[2*index+1]+closed[j]%2;

      if (!(d.getocc(a) || d.getocc(b))) {
	dets.push_back(d);
	Determinant& di = *dets.rbegin();
	di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);
      }
    }
  }
  return;
}





void SHCIgetdeterminants::getDeterminantsStochastic(Determinant& d, double epsilon, CItype ci1, CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<CItype>& numerator1, vector<double>& numerator2, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

  double Nmcd = 1. * Nmc;
  int norbs = d.norbs;
  //char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0);
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  //d.getRepArray(detArray);

  double Energyd = d.Energy(int1, int2, coreE);


  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    //if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
#ifndef Complex 
    if (closed[i]%2 != open[a]%2 || irreps[closed[i]/2] != irreps[open[a]/2]) continue;
#endif
    CItype integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);


    if (abs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);

      numerator1.push_back(integral*ci1);
#ifndef Complex
      numerator2.push_back( integral*integral*ci1*(ci1*Nmcd/(Nmcd-1)- ci2));
#else
      numerator2.push_back( (integral*integral*ci1*(ci1*Nmcd/(Nmcd-1)- ci2)).real());
#endif
      energy.push_back(E);
    }
  }

  if (abs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X*(X+1)/2+Y;
    size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
    double* integrals = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
    int* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;


    for (size_t index=start; index<end; index++) {
      if (abs(integrals[index]) <epsilon) break;
      int a = 2* orbIndices[2*index] + closed[i]%2, b= 2*orbIndices[2*index+1]+closed[j]%2;

      if (!(d.getocc(a) || d.getocc(b))) {
	dets.push_back(d);
	Determinant& di = *dets.rbegin();
	di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);

	double sgn = 1.0;
	di.parity(a, b, closed[i], closed[j], sgn);

	numerator1.push_back(integrals[index]*sgn*ci1);
#ifndef Complex
	numerator2.push_back( integrals[index]*integrals[index]*ci1*(ci1*Nmcd/(Nmcd-1)- ci2));
#else
	numerator2.push_back( (integrals[index]*integrals[index]*ci1*(ci1*Nmcd/(Nmcd-1)- ci2)).real());
#endif
	double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);
	energy.push_back(E);

      }
    }

  }
  return;
}


void SHCIgetdeterminants::getDeterminantsStochastic2Epsilon(Determinant& d, double epsilon, double epsilonLarge, CItype ci1, CItype ci2, oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<CItype>& numerator1A, vector<CItype>& numerator2A, vector<char>& present, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

  double Nmcd = 1.*Nmc;
  int norbs = d.norbs;
  //char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0);
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  //d.getRepArray(detArray);

  double Energyd = d.Energy(int1, int2, coreE);


  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (closed[i]%2 != open[a]%2 || irreps[closed[i]/2] != irreps[open[a]/2]) continue;
    CItype integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);


    if (abs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);

      numerator1A.push_back(integral*ci1);
#ifndef Complex
      numerator2A.push_back( integral*integral*ci1 *(ci1*Nmcd/(Nmcd-1)- ci2));
#else
      numerator2A.push_back( (integral*integral*ci1 *(ci1*Nmcd/(Nmcd-1)- ci2)).real() );
#endif
      if (abs(integral) >epsilonLarge)
	present.push_back(true);
      else
	present.push_back(false);

      energy.push_back(E);
    }
  }

  if (abs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    int X = max(I, J), Y = min(I, J);

    int pairIndex = X*(X+1)/2+Y;
    size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
    double* integrals = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
    int* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs : I2hb.oppositeSpinPairs;


    for (size_t index=start; index<end; index++) {
      if (abs(integrals[index]) <epsilon) break;
      int a = 2* orbIndices[2*index] + closed[i]%2, b= 2*orbIndices[2*index+1]+closed[j]%2;

      if (!(d.getocc(a) || d.getocc(b))) {
	dets.push_back(d);
	Determinant& di = *dets.rbegin();
	di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);

	double sgn = 1.0;
	di.parity(a, b, closed[i], closed[j], sgn);

	numerator1A.push_back(integrals[index]*sgn*ci1);
#ifndef Complex
	numerator2A.push_back( integrals[index]*integrals[index]*ci1*(ci1*Nmcd/(Nmcd-1)- ci2));
#else
	numerator2A.push_back( (integrals[index]*integrals[index]*ci1*(ci1*Nmcd/(Nmcd-1)- ci2)).real());
#endif
	//numerator2A.push_back( abs(integrals[index]*integrals[index]*ci1*(ci1*Nmc/(Nmc-1)- ci2)));


	if (abs(integrals[index]) >epsilonLarge)
	  present.push_back(true);
	else
	  present.push_back(false);

	double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);
	energy.push_back(E);

      }

    }
  }
  return;
}

