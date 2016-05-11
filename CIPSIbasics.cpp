#include "Determinants.h"
#include "CIPSIbasics.h"
#include "integral.h"
#include <vector>
#include "math.h"
#include "Hmult.h"
using namespace std;

void getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, double coreE, double E0, std::vector<Determinant>& dets) {

  int norbs = d.norbs;
  char open[norbs], closed[norbs], detArray[norbs], diArray[norbs];
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
  for (int a=0; a<nopen; a++){
    for (int b=0; b<a; b++){
      for (int i=0; i<nclosed; i++) {
	for (int j=0; j<i; j++) {

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
