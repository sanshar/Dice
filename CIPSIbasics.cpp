#include "Determinants.h"
#include "CIPSIbasics.h"
#include "integral.h"
#include <vector>
#include "math.h"
#include "Hmult.h"
using namespace std;
using namespace Eigen;

void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::vector<Determinant>& dets) {
  
  int norbs = d.norbs;
  unsigned short open[norbs], closed[norbs];char detArray[norbs], diArray[norbs];
  int nclosed = d.getOpenClosed(open, closed);
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);
  
  for (int a=0; a<nopen; a++){
    for (int i=0; i<nclosed; i++) {
      double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);
      if (fabs(integral) > epsilon ) {
	Determinant di = d;
	di.setocc(open[a], true); di.setocc(closed[i],false);
	//di.getRepArray(diArray);
	//if (abs(integral/(E0-Energy(diArray,norbs, int1,int2,coreE))) >epsilon)
	dets.push_back(di);
      }
    }
  }

  if (fabs(int2.maxEntry) <epsilon) return;
  for (int i=0; i<nclosed; i++) {
    for (int j=0; j<i; j++) {
      int I = closed[i]/2, J = closed[j]/2;
      if (fabs(2*int2.maxEntryPerPair(I,J) ) < epsilon) continue;
      //if (fabs(int2.maxEntryPerPair[A*(A+1)/2+I] ) < epsilon) continue;
      for (int a=0; a<nopen; a++){
	for (int b=0; b<a; b++){
	  
	  double integral = abs(int2(closed[i],open[a],closed[j],open[b]) - int2(closed[i],open[b],closed[j],open[a]));
	  
	  if (fabs(integral) > epsilon ) {
	    Determinant di = d;
	    di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);
	    //di.getRepArray(diArray);
	    //if  (fabs(integral/(E0-Energy(diArray, norbs, int1,int2,coreE))) >epsilon)
	    dets.push_back(di);
	  }
	}
      }
    }
  }
  return;
}
  
  
void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::set<Determinant>& dets) {
  
  int norbs = d.norbs;
  unsigned short open[norbs], closed[norbs]; char detArray[norbs], diArray[norbs];
  int nclosed = d.getOpenClosed(open, closed);
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);
  
  std::vector<std::vector<Determinant> > thrdDeterminants(omp_get_max_threads());
  //#pragma omp parallel for schedule(dynamic)
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);
    if (fabs(integral) > epsilon ) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      //thrdDeterminants[omp_get_thread_num()].push_back(di);
      dets.insert(di);
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {
    int i=ij/nclosed, j = ij%nclosed;
    if (i<= j) continue;
    //int A = max(a,i), I = min(a,i);
    int I = closed[i]/2, J = closed[j]/2;
    if (fabs(2*int2.maxEntryPerPair(I,J) ) < epsilon) continue;
    for (int a=0; a<nopen; a++){
      for (int b=0; b<a; b++){
	
	double integral = abs(int2(closed[i],open[a],closed[j],open[b]) - int2(closed[i],open[b],closed[j],open[a]));
	
	if (fabs(integral) > epsilon ) {
	  Determinant di = d;
	  di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);

	  //thrdDeterminants[omp_get_thread_num()].push_back(di);
	  dets.insert(di);
	}
      }
      
    }
  }

  //for (int thrd=0; thrd<omp_get_max_threads(); thrd++)
  //for (int i=0; i<thrdDeterminants[thrd].size(); i++)
  //dets.insert(thrdDeterminants[thrd][i]);

  return;
}

void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::list<Determinant>& dets) {
  
  int norbs = d.norbs;
  unsigned short open[norbs], closed[norbs]; char detArray[norbs], diArray[norbs];
  int nclosed = d.getOpenClosed(open, closed);
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);
  
  for (int a=0; a<nopen; a++){
    for (int i=0; i<nclosed; i++) {
      double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);
      if (fabs(integral) > epsilon ) {
	Determinant di = d;
	di.setocc(open[a], true); di.setocc(closed[i],false);
	//di.getRepArray(diArray);
	//if (abs(integral/(E0-Energy(diArray,norbs, int1,int2,coreE))) >epsilon)
	dets.push_back(di);
      }
    }
  }
  if (fabs(int2.maxEntry) <epsilon) return;
    for (int i=0; i<nclosed; i++) { 
      for (int j=0; j<i; j++) {
	//int A = max(a,i), I = min(a,i);
	int I = closed[i]/2, J = closed[j]/2;
	if (fabs(2*int2.maxEntryPerPair(I,J) ) < epsilon) continue;
	for (int a=0; a<nopen; a++){
	  for (int b=0; b<a; b++){
	  
	  double integral = abs(int2(closed[i],open[a],closed[j],open[b]) - int2(closed[i],open[b],closed[j],open[a]));
	  
	  if (fabs(integral) > epsilon ) {
	    Determinant di = d;
	    di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);
	    //di.getRepArray(diArray);
	    //if  (fabs(integral/(E0-Energy(diArray, norbs, int1,int2,coreE))) >epsilon)
	    dets.push_back(di);
	  }
	}
	
      }
    }
  }
  return;
}



void getDeterminants(std::vector<Determinant>& detsIn, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, set<Determinant>& dets,MatrixXd& ci) {

  int norbs = Determinant::norbs;
  for (int dindex=0;dindex<detsIn.size(); dindex++) {
    Determinant& d = detsIn[dindex];
    unsigned short open[norbs], closed[norbs];char detArray[norbs], diArray[norbs];
    int nclosed = d.getOpenClosed(open, closed);
    int nopen = norbs-nclosed;

    d.getRepArray(detArray);
    for (int a=0; a<nopen; a++){
      for (int i=0; i<nclosed; i++) {
	double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);
	if (fabs(integral) > fabs(epsilon/ci(dindex,0)) ) {
	  Determinant di = d;
	  di.setocc(open[a], true); di.setocc(closed[i],false);
	  //di.getRepArray(diArray);
	  //if (abs(integral/(E0-Energy(diArray,norbs, int1,int2,coreE))) >epsilon)
	  dets.insert(di);
	}
      }
    }
  }

  if (fabs(int2.maxEntry) <epsilon) return;
  for (int i=0; i<norbs; i++) 
    for (int j=0; j<i; j++) {

      //int I = i/2, J = j/2;
      //if (fabs(2*int2.maxEntryPerPair(I,J) ) < epsilon) continue;
      for (int a=0; a<norbs; a++){
	for (int b=0; b<a; b++){
	  
	  double integral = abs(int2(i,a,j,b) - int2(i,b,j,a));
	  if (fabs(integral) > epsilon ) {
	    for (int dindex=0;dindex<detsIn.size(); dindex++) {
	      Determinant& d = detsIn[dindex];
	      if (d.getocc(i) && d.getocc(j) && !d.getocc(a) && !d.getocc(b) && fabs(integral) > fabs(epsilon/ci(dindex,0))) {
		Determinant di = d;
		di.setocc(a, true), di.setocc(b, true), di.setocc(i,false), di.setocc(j, false);
		dets.insert(di);
	      }
	    }
	  }
	}
      }
  }
  return;
}


