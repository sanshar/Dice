#include <iostream>
#include "global.h"
#include "Determinants.h"
#include "integral.h"
#include "Hmult.h"
#include "CIPSIbasics.h"
#include "Davidson.h"
#include <Eigen/Dense>

using namespace Eigen;
int Determinant::norbs = 1; //spin orbitals

int main(int argc, char* argv[]) {
  std::cout.precision(15);
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE;
  readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE);
  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals

  //make HF determinant
  Determinant d;
  d.setocc(0,true); d.setocc(1,true); d.setocc(4,true); d.setocc(5,true);
  d.setocc(6,true); d.setocc(7,true); d.setocc(8,true); d.setocc(9,true);

  char detchar[norbs]; d.getRepArray(detchar);
  std::cout << Energy(detchar,norbs,I1,I2,coreE)<<" "<<coreE<<std::endl;

  char closed[nelec], open[norbs-nelec];
  int o = d.getOpenClosed(open, closed); int v=norbs-o;
  std::vector<Determinant> dets(o*(o-1)*v*(v-1)/4+1+o*v);
  dets[0] = d;


  int index = 1;

  for (int a=0; a<v; a++){
    for (int i=0; i<o; i++) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      dets[index]=di;
      index++;
    }
  }
  for (int a=0; a<v; a++){
    for (int b=0; b<v; b++){
      if (b >= a) continue;
      for (int i=0; i<o; i++) {
	for (int j=0; j<o; j++) {
	  if (j >= i) continue;
	  Determinant di = d;
	  di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);
	  dets[index] = di;
	  index++;
	}
      }
    }
  }

  if (false) {
    MatrixXd Ham(dets.size(), dets.size());
    char deti[norbs], detj[norbs];
    for (int i=0; i<dets.size(); i++) {
      dets[i].getRepArray(deti); 
      for (int j=0; j<dets.size(); j++) {
	dets[j].getRepArray(detj);
	Ham(i,j) = Hij(deti, detj, norbs, I1, I2, coreE);
	if (i != j) Ham(j,i) = Ham(i,j);
      }
    }
    
    SelfAdjointEigenSolver<MatrixXd> eigensolver(Ham);
    if (eigensolver.info() != Success) abort();
    cout << "The eigenvalues of Ham are:\n" << eigensolver.eigenvalues()[0] << endl;
  }
  else {
    char detChar[norbs*dets.size()]; 
    MatrixXd X0(dets.size(), 1); X0 *= 0.0; X0(0,0) = 1.0;
    MatrixXd diag(dets.size(), 1); diag *= 0.0;
    for (int k=0; k<dets.size(); k++) {
      dets[k].getRepArray(detChar+norbs*k);
      diag(k,0) = Energy(detChar+norbs*k, norbs, I1, I2, coreE);
    }
    Hmult H(detChar, norbs, I1, I2, coreE);
    davidson(H, X0, diag, 5, 1e-10);
  }
  return 0;
}
