#include "interface.h"


BasisSet basis;

double doubleFact(size_t n) {
  if (n == -1) return 1.;
  double val = 1.0;
  int limit = n/2;
  for (int i=0; i<limit; i++)
    val *= (n-2*i);
  return val;
}

void initPeriodic(int* pshls, int *patm, int pnatm,
                  int *pbas, int pnbas, double *penv,
                  double* lattice) {
  basis.BasisShells.resize(pnbas);
  for (int i=0; i<pnbas; i++) {
    BasisShell& shell = basis.BasisShells[i];
    int atm = pbas[8*i];

    //position
    shell.Xcoord = penv[patm[6*atm+1]+0];
    shell.Ycoord = penv[patm[6*atm+1]+1];
    shell.Zcoord = penv[patm[6*atm+1]+2];

    shell.l  = pbas[8*i + 1];
    int nFn = pbas[8*i + 2], nCo = pbas[8*i + 3];
    
    shell.nFn = nFn;
    shell.nCo = nCo;

    assert(pbas[8*i+6] - pbas[8*i+5] == shell.nFn);
    
    shell.exponents.resize(nFn);
    shell.contractions.resize(nFn, nCo);
    for (int f=0; f<nFn; f++) {
      shell.exponents[f] = penv[ pbas[8*i + 5] + f];
    }

    //the normalization is really weird coming from pyscf
    double norm = sqrt(2*shell.l + 1) *  pow(1./M_PI/4., 0.5);// * sqrt(4* M_PI/ (2*LA+1));
    
    for (int f=0; f<nFn; f++)
      for (int co=0; co<nCo; co++) {
        shell.contractions(f, co) = penv[ pbas[8*i+5] + (co+1) * nFn + f] * norm;
      }
  }
}

