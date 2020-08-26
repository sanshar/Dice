#include <iostream>
#include <cmath>
#include <string.h>
#include <algorithm>
#include <numeric> 
#include <complex>

#include "CxMemoryStack.h"
#include "GeneratePolynomials.h"
#include "IrAmrr.h"
#include "IrBoysFn.h"
#include "BasisShell.h"
#include "Integral2c_Boys.h"
#include "LatticeSum.h"
#include "interface.h"

using namespace std;
using namespace ir;


void Add2(double * pOut, double const * pIn, double f, size_t n)
{
  size_t i = 0;
  for ( ; i < (n & ~3); i += 4 ) {
    pOut[i]   += f * pIn[i];
    pOut[i+1] += f * pIn[i+1];
    pOut[i+2] += f * pIn[i+2];
    pOut[i+3] += f * pIn[i+3];
  }
  pOut += i;
  pIn += i;
  switch(n - i) {
    case 3: pOut[2] += f*pIn[2];
    case 2: pOut[1] += f*pIn[1];
    case 1: pOut[0] += f*pIn[0];
    default: break;
  }

}



// output: contracted kernels Fm(rho,T), format: (TotalL+1) x nCoA x nCoC
void Int2e2c_EvalCoKernels(double *pCoFmT, uint TotalL,
                           BasisShell *pA, BasisShell *pC,
                           double Tx, double Ty, double Tz,
                           double PrefactorExt, double* pInv2Alpha, double* pInv2Gamma,
                           Kernel* kernel,
                           LatticeSum& latsum, ct::FMemoryStack &Mem)
{  
  double t = Tx*Tx + Ty*Ty + Tz*Tz;
  double *pFmT;
  Mem.Alloc(pFmT, TotalL + 1); // FmT for current primitive.

  double Eta2Rho = kernel->getname() == coulombKernel ? latsum.Eta2RhoCoul : latsum.Eta2RhoOvlp;
  
  // loop over primitives (that's all the per primitive stuff there is)
  for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
  for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
  {
    double
        Alpha = pA->exponents[iExpA],
        Gamma = pC->exponents[iExpC],
        InvEta = 1./(Alpha + Gamma),
        Rho = (Alpha * Gamma)*InvEta, // = (Alpha * Gamma)*/(Alpha + Gamma)
        Prefactor = (M_PI*InvEta)*std::sqrt(M_PI*InvEta); // = (M_PI/(Alpha+Gamma))^{3/2}
    
    Prefactor *= PrefactorExt;
    Prefactor *= pInv2Gamma[iExpC] * pInv2Alpha[iExpA];
    
    
    
    //If eta is smaller than desiredrho then no contribution from real space
    double eta = 1.0;
    if (Rho <= Eta2Rho) continue;
    else {
      eta = sqrt(Eta2Rho/Rho);
    }

    if (Rho > Eta2Rho &&  kernel->getname() != coulombKernel) eta = 1.0;

    // calculate derivatives (D/Dt)^m exp(-rho t) with t = (A-C)^2.
    kernel->getValueRSpace(pFmT, Rho*t, TotalL, Prefactor, Rho, eta, Mem);
    
    // convert from Gm(rho,T) to Fm(rho,T) by absorbing powers of rho
    // (those would normally be present in the R of the MDRR)
    double
        RhoPow = 1.;
    double maxVal = 0;
    for ( uint i = 0; i < TotalL + 1; ++ i ){
      pFmT[i] *= RhoPow;
      RhoPow *= -2*Rho;
      if (maxVal < abs(pFmT[i])) maxVal = abs(pFmT[i]);
    }

    if (maxVal*max(1., pow(t,TotalL/2)) < latsum.screen) continue;
    
    // contract (lamely). However, normally either nCo
    // or nFn, or TotalL (or even all of them at the same time)
    // will be small, so I guess it's okay.
    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
    for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
      double CoAC = pC->contractions(iExpC, iCoC) *
          pA->contractions(iExpA, iCoA);
      Add2(&pCoFmT[(TotalL+1)*(iCoA + pA->nCo*iCoC)],
           pFmT, CoAC, (TotalL+1));
    }
  }
  Mem.Free(pFmT);
}


void makeRealSummation(double *&pOutR, unsigned &TotalCo, BasisShell *pA, BasisShell *pC,
                       double Tx, double Ty, double Tz, double Prefactor,   unsigned TotalLab,
                       double* pInv2Alpha, double* pInv2Gamma, Kernel* kernel,
                       LatticeSum& latsum, ct::FMemoryStack& Mem) {
  
  double* pOutR_loc, *pCoFmT, *pDataR_LapC;

  int L = TotalLab +  (kernel->getname() == kineticKernel ? 2 : 0);

  Mem.ClearAlloc(pOutR_loc, (TotalLab+1)*(TotalLab+2)/2 * TotalCo);
  Mem.ClearAlloc(pDataR_LapC, (L+1)*(L+2)/2);
  
  size_t *Tidx; //indexed such that T will increase
  latsum.getIncreasingIndex(Tidx, Tx, Ty, Tz, Mem);
  
  for (int r = 0; r<latsum.Rdist.size(); r++) {

    double Tx_r = Tx + latsum.Rcoord[3*Tidx[r]+0],
           Ty_r = Ty + latsum.Rcoord[3*Tidx[r]+1],
           Tz_r = Tz + latsum.Rcoord[3*Tidx[r]+2];

    
    Mem.ClearAlloc(pCoFmT, (L+1) * TotalCo);
    Int2e2c_EvalCoKernels(pCoFmT, L, pA, pC, Tx_r, Ty_r, Tz_r, Prefactor,
                          pInv2Alpha, pInv2Gamma, kernel, latsum, Mem);


    //go from [0]^m -> [r]^0 using mcmurchie-davidson
    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
    for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {

      double
          *pFmT = &pCoFmT[(L+1) * (iCoA + pA->nCo*iCoC)],
          *pDataR_ = &pOutR_loc[nCartY(TotalLab) * (iCoA + pA->nCo*iCoC)];
    
      if (kernel->getname() == kineticKernel) {
         ShellMdrr(pDataR_LapC, pFmT, Tx_r, Ty_r, Tz_r, L);
         ShellLaplace(pDataR_, pDataR_LapC, 1, L - 2);
      }
      else 
        ShellMdrr(pDataR_, pFmT, Tx_r, Ty_r, Tz_r, L);
    }

    double maxPout = 0;
    for (int i=0; i<(TotalLab+1)*(TotalLab+2)/2 * TotalCo; i++) {
      pOutR[i] += pOutR_loc[i];
      maxPout = max(maxPout, abs(pOutR_loc[i]));
      pOutR_loc[i] = 0.0; 
    }
    
    Mem.Free(pCoFmT);
    if (maxPout < latsum.screen) {
      break;
    }
  }
}

void makeReciprocalSummation(double *&pOutR, unsigned &TotalCo, BasisShell *pA, BasisShell *pC, double Tx, double Ty, double Tz, double Prefactor,   unsigned TotalLab, double* pInv2Alpha, double* pInv2Gamma, Kernel* kernel, LatticeSum& latsum, ct::FMemoryStack& Mem)
{
  int L = TotalLab ;

  int LA = pA->l, LC = pC->l;
  
  //calcualte the reciprocal space contribution
  double background ;   
  double *pReciprocalSum;
  Mem.Alloc(pReciprocalSum, (L+1)*(L+2)/2);

  double Eta2Rho = kernel->getname() == coulombKernel ? latsum.Eta2RhoCoul : latsum.Eta2RhoOvlp;
  double screen = latsum.screen;

  //this will work for all rho that are above the Et2Rho in coulomb kernel

  ksumTime1.start();

  if (kernel->getname() == coulombKernel) {
    //for (int i=0; i<(L+1)*(L+2)/2; i++)
    //pReciprocalSum[i] = 0.0 ;
    ksumKsum.start();
    int T1 = latsum.indexCenter(*pA);
    int T2 = latsum.indexCenter(*pC);
    int T = T1 == T2 ? 0 : max(T1, T2)*(max(T1, T2)+1)/2 + min(T1, T2);
    int startIdx = latsum.KSumIdx[T][L];
    for (int i=0; i<(L+1)*(L+2)/2; i++)
      pReciprocalSum[i] = latsum.KSumVal[startIdx+i] ;

    ksumKsum.stop();
    
    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
    for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
      double CoAC = 0;
      double bkgrnd = 0.0;
      for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
      for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
      {
        double
            Alpha = pA->exponents[iExpA],
            Gamma = pC->exponents[iExpC],
            InvEta = 1./(Alpha + Gamma),
            Rho = (Alpha * Gamma)*InvEta; 
        
        if (Rho <= Eta2Rho) continue;
        
        double Eta = sqrt(Eta2Rho/Rho);
        double Omega = Rho <= Eta2Rho ? 1.0e9 : sqrt(Rho * Eta * Eta /(1. - Eta * Eta)) ; 
        double Eta2rho = Eta2Rho;
        
        double scale = Prefactor * pInv2Gamma[iExpC] * pInv2Alpha[iExpA];
        scale *= M_PI*M_PI*M_PI*M_PI/pow(Alpha*Gamma, 1.5)/ latsum.RVolume;
        
        CoAC += pC->contractions(iExpC, iCoC) *
            pA->contractions(iExpA, iCoA) * scale;
        
        if (LA == 0 && LC == 0)
          bkgrnd +=   pC->contractions(iExpC, iCoC) *
            pA->contractions(iExpA, iCoA) *
              M_PI * 16.*M_PI*M_PI * tgamma((LA+3)/2.)/(2. * pow(Alpha,0.5*(LA+3)))
              * tgamma((LC+3)/2.)/(2. * pow(Gamma,0.5*(LC+3)))/Omega/Omega/ latsum.RVolume;
        
      }

      Add2(&pOutR[(TotalLab+1)*(TotalLab+2)/2*(iCoA + pA->nCo*iCoC)],
           pReciprocalSum, CoAC, (TotalLab+1)*(TotalLab+2)/2);
      if (LA == 0 && LC == 0)
        Add2(&pOutR[(TotalLab+1)*(TotalLab+2)/2*(iCoA + pA->nCo*iCoC)],
             &bkgrnd, -1., (TotalLab+1)*(TotalLab+2)/2);

    }
  }

  ksumTime1.stop();
  
  ksumTime2.start();
  //now go back to calculating contribution from reciprocal space
  for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
  for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
  {
    double
        Alpha = pA->exponents[iExpA],
        Gamma = pC->exponents[iExpC],
        InvEta = 1./(Alpha + Gamma),
        Rho = (Alpha * Gamma)*InvEta; // = (Alpha * Gamma)*/(Alpha + Gamma)
    
    //if (Rho < latsum.Kdist[0]/120) continue;
    
    double Eta = Rho <= Eta2Rho ? 1. : sqrt(Eta2Rho/Rho);
    double Omega = Rho <= Eta2Rho ? 1.0e9 : sqrt(Rho * Eta * Eta /(1. - Eta * Eta)) ; 
    double Eta2rho = Eta * Eta * Rho;

    
    //coulomb kernel always includes reciprocal space summations
    //if (Rho > Eta2Rho && kernel->getname() != coulombKernel) continue; 
    if (Rho > Eta2Rho ) continue; 
    else if (Rho <= Eta2Rho &&  kernel->getname() != coulombKernel) Eta2rho = Rho;

    double scale = Prefactor * pInv2Gamma[iExpC] * pInv2Alpha[iExpA];    
    if (kernel->getname() == coulombKernel)
      scale *= M_PI*M_PI*M_PI*M_PI/pow(Alpha*Gamma, 1.5)/ latsum.RVolume; 
    else if (kernel->getname() != coulombKernel)
      scale *= pow(M_PI*M_PI/Alpha/Gamma, 1.5)/latsum.RVolume;
    


    for (int i=0; i<(L+1)*(L+2)/2; i++)
      pReciprocalSum[i] = 0.0;
    
    
    //this is ugly, in coulomb kernel the G=0 term is discarded but not in others
    if (kernel->getname() == overlapKernel) {
      double expVal = kernel->getValueKSpace(0, 1.0, Eta2rho);
      double maxG = getHermiteReciprocal(L, pReciprocalSum,
                                         0., 0., 0., Tx, Ty, Tz, expVal, scale);
    }
    
    for (int k=0; k<latsum.Kdist.size(); k++) {
      double expVal = kernel->getValueKSpace(latsum.Kdist[k], 1.0, Eta2rho);
      
      double maxG = getHermiteReciprocal(L, pReciprocalSum,
                                         latsum.Kcoord[3*k+0],
                                         latsum.Kcoord[3*k+1], latsum.Kcoord[3*k+2],
                                         Tx, Ty, Tz,
                                         expVal, scale);
      
      if (abs(maxG * scale * expVal) < screen ) {
        break;
      }
    }      
    
    
    
    //the background term, only applies to coulomb kernel
    if (!(Rho <= Eta2Rho) && LA == 0 && LC == 0 && kernel->getname() == coulombKernel) {
      pReciprocalSum[0] -= M_PI * 16.*M_PI*M_PI * tgamma((LA+3)/2.)/(2. * pow(Alpha,0.5*(LA+3)))
          * tgamma((LC+3)/2.)/(2. * pow(Gamma,0.5*(LC+3)))/Omega/Omega/ latsum.RVolume;
    }
    
    //cout <<"bkgrnd "<< pReciprocalSum[0]<<endl;
    
    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
    for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
      double CoAC = pC->contractions(iExpC, iCoC) *
          pA->contractions(iExpA, iCoA);

      Add2(&pOutR[(TotalLab+1)*(TotalLab+2)/2*(iCoA + pA->nCo*iCoC)],
           pReciprocalSum, CoAC, (TotalLab+1)*(TotalLab+2)/2);
    }
  }
  ksumTime2.stop();

  
  Mem.Free(pReciprocalSum);
}



void Int2e2c_EvalCoShY(double *&pOutR, unsigned &TotalCo, BasisShell *pA, BasisShell *pC, double Prefactor,   unsigned TotalLab, double* pInv2Alpha, double* pInv2Gamma, Kernel* kernel, LatticeSum& latsum, ct::FMemoryStack& Mem)
{
  //CHANGE THIS to minimum distance between A and periodic image of C
  double Tx, Ty, Tz;
  latsum.getRelativeCoords(pA, pC, Tx, Ty, Tz);
  
  TotalCo = pA->nCo * pC->nCo;
  
  unsigned
      L = TotalLab ;

  Mem.ClearAlloc(pOutR, (L+1)*(L+2)/2 * TotalCo);
  
  realSumTime.start();
  makeRealSummation(pOutR, TotalCo, pA, pC, Tx, Ty, Tz, Prefactor, TotalLab, pInv2Alpha, pInv2Gamma, kernel, latsum, Mem);
  realSumTime.stop();

  kSumTime.start();
  Tx = pA->Xcoord - pC->Xcoord;
  Ty = pA->Ycoord - pC->Ycoord;
  Tz = pA->Zcoord - pC->Zcoord;
  makeReciprocalSummation(pOutR, TotalCo, pA, pC, Tx, Ty, Tz, Prefactor, TotalLab, pInv2Alpha, pInv2Gamma, kernel, latsum, Mem);
  kSumTime.stop();

  
}

void EvalInt2e2c( double *pOut, size_t StrideA, size_t StrideC,
                  BasisShell *pA, BasisShell *pC, double Prefactor, bool Add,
                  Kernel* kernel,
                  LatticeSum& latsum, ct::FMemoryStack &Mem )
{
   uint
      lc = pC->l, la = pA->l,
      TotalCo;

   //we will need these in both the real and reciprocal space summations
   double
       *pInv2Alpha, *pInv2Gamma;
   Mem.Alloc(pInv2Alpha, pA->nFn);
   Mem.Alloc(pInv2Gamma, pC->nFn);  
   for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
     pInv2Alpha[iExpA] = bool(pA->l)? std::pow(+1.0/(2*pA->exponents[iExpA]), (int)pA->l) : 1.;
   for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
     pInv2Gamma[iExpC] = bool(pC->l)? std::pow(-1.0/(2*pC->exponents[iExpC]), (int)pC->l) : 1.;
  

   double
      *pDataR, *pR1, *pFinal;
   
   Int2e2c_EvalCoShY(pDataR, TotalCo, pA, pC, Prefactor, la + lc, pInv2Alpha, pInv2Gamma, kernel, latsum, Mem);

   Mem.Alloc(pR1, nCartY(la)*(2*lc+1) * TotalCo);
   Mem.Alloc(pFinal, (2*la+1)*(2*lc+1) * TotalCo);

   ShTrA_YY(pR1, pDataR, lc, (la + lc), TotalCo);
   ShTrA_YY(pFinal, pR1, la, la, (2*lc + 1)*TotalCo);
   // now: (2*la+1) x (2*lc+1) x nCoA x nCoC

   Scatter2e2c(pOut, StrideA, StrideC, pFinal, la, lc, 1, pA->nCo, pC->nCo, Add);

   Mem.Free(pInv2Alpha);
}

// write (2*la+1) x (2*lc+1) x nCoA x nCoC matrix to final destination.
void Scatter2e2c(double * pOut, size_t StrideA, size_t StrideC,
                 double const * pIn, size_t la, size_t lc,
                 size_t nComp, size_t nCoA, size_t nCoC, bool Add)
{
   size_t nShA = 2*la+1, nShC = 2*lc+1;
   if ( Add ) {
      for (size_t iCoC = 0; iCoC < nCoC; ++ iCoC)
         for (size_t iCoA = 0; iCoA < nCoA; ++ iCoA)
            for (size_t iShC = 0; iShC < nShC; ++ iShC)
               for (size_t iShA = 0; iShA < nShA; ++ iShA)
                  pOut[(iShA + nShA*iCoA)*StrideA + (iShC + nShC*iCoC)*StrideC]
                     += pIn[iShA + nShA * (iShC + nShC * nComp * (iCoA + nCoA * iCoC))];
   } else {
      for (size_t iCoC = 0; iCoC < nCoC; ++ iCoC)
         for (size_t iCoA = 0; iCoA < nCoA; ++ iCoA)
            for (size_t iShC = 0; iShC < nShC; ++ iShC)
               for (size_t iShA = 0; iShA < nShA; ++ iShA)
                  pOut[(iShA + nShA*iCoA)*StrideA + (iShC + nShC*iCoC)*StrideC]
                      = pIn[iShA + nShA * (iShC + nShC * nComp *(iCoA + nCoA * iCoC))];
   }
}
