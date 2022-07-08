#include <cmath>
#include <iostream>

#include "Kernel.h"
#include "IrBoysFn.h"

using namespace std;
using namespace ir;

void CoulombKernel::getValueRSpace(double* pFmT, double T, int L, double factor, double rho,
                                   double eta, ct::FMemoryStack& Mem) {
  IrBoysFn(pFmT, T, L, factor*(2*M_PI)/rho);

  double* pFmtOmega;
  Mem.Alloc(pFmtOmega, L+1);
  IrBoysFn(pFmtOmega, eta*eta*T, L, factor*(2*M_PI)/rho);

  double etaPow = eta;
  for (int i=0; i<L+1; i++) {
    pFmT[i] -= pFmtOmega[i] * etaPow;
    etaPow *= eta * eta;
  }
  //cout << rho<<"  "<<eta<<"  "<<pFmT[0]<<"  "<<pFmtOmega[0]<<endl;
  Mem.Free(pFmtOmega);

}

inline double CoulombKernel::getValueKSpace(double Gsq, double factor, double rho){
  return exp(-Gsq/rho/4.0)/(Gsq/4) * factor;
}

void OverlapKernel::getValueRSpace(double* pFmT, double T, int L, double factor, double rho,
                                   double eta, ct::FMemoryStack& Mem) {
  double val = exp(-T)*factor;// * pow(M_PI/rho, 1.5);
  for (int i=0; i<L+1; i++) {
    pFmT[i] = val;
  }
}

inline double OverlapKernel::getValueKSpace(double Gsq, double factor, double rho) {
  return exp(-Gsq/rho/4.0) * factor;
}


void KineticKernel::getValueRSpace(double* pFmT, double T, int L, double factor, double rho,
                                   double eta, ct::FMemoryStack& Mem) {
  double val = -0.5*exp(-T)*factor;// * pow(M_PI/rho, 1.5);
  //double val = -0.5*exp(-T)*(-6*rho + 4*T)*factor;
  for (int i=0; i<L+1; i++) {
    pFmT[i] = val;
  }
}

inline double KineticKernel::getValueKSpace(double Gsq, double factor, double rho) {
  return 0.5*exp(-Gsq/rho/4.0) * Gsq * factor;
}
