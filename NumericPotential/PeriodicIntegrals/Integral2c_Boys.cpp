#include <iostream>
#include <cmath>
#include <string.h>
#include <algorithm>
#include <numeric> 

#include "CxMemoryStack.h"
#include "GeneratePolynomials.h"
#include "IrAmrr.h"
#include "IrBoysFn.h"
#include "BasisShell.h"
#include "Integral2c_Boys.h"

using namespace std;
using namespace ir;

double dotProduct(double* vA, double* vB) {
  return vA[0] * vB[0] + vA[1] * vB[1] + vA[2] * vB[2];
};

void crossProduct(double* v_A, double* v_B, double* c_P, double factor=1.) {
  c_P[0] = factor * (v_A[1] * v_B[2] - v_A[2] * v_B[1]);
  c_P[1] = -factor * (v_A[0] * v_B[2] - v_A[2] * v_B[0]);
  c_P[2] = factor * (v_A[0] * v_B[1] - v_A[1] * v_B[0]);
};

double getKLattice(double* KLattice, double* Lattice) {
  vector<double> cross(3);
  crossProduct(&Lattice[3], &Lattice[6], &cross[0]);
  double Volume = dotProduct(&Lattice[0], &cross[0]);
  
  crossProduct(&Lattice[3], &Lattice[6], &KLattice[0], 2*M_PI/Volume);
  crossProduct(&Lattice[6], &Lattice[0], &KLattice[3], 2*M_PI/Volume);
  crossProduct(&Lattice[0], &Lattice[3], &KLattice[6], 2*M_PI/Volume);
  return Volume;
};

double dist(double a, double b, double c) {
  return a*a + b*b + c*c;
}

void getRelativeCoords(BasisShell *pA, BasisShell *pC, LatticeSum& latsum,
                       double& Tx, double& Ty, double& Tz) {
  double Txmin = pA->Xcoord - pC->Xcoord,
      Tymin = pA->Ycoord - pC->Ycoord,
      Tzmin = pA->Zcoord - pC->Zcoord;
  Tx = Txmin; Ty = Tymin; Tz = Tzmin;

  if (Txmin == 0 && Tymin == 0 && Tzmin == 0) {
    return;
  }
  
  for (int nx=-1; nx<=1; nx++)
  for (int ny=-1; ny<=1; ny++)
  for (int nz=-1; nz<=1; nz++)
  {
    if (dist(Tx + nx * latsum.RLattice[0] + ny * latsum.RLattice[1] + nz * latsum.RLattice[2],
             Ty + nx * latsum.RLattice[3] + ny * latsum.RLattice[4] + nz * latsum.RLattice[5],
             Tz + nx * latsum.RLattice[6] + ny * latsum.RLattice[7] + nz * latsum.RLattice[8])
        < dist(Txmin, Tymin, Tzmin))  {
      Txmin = Tx + nx * latsum.RLattice[0] + ny * latsum.RLattice[1] + nz * latsum.RLattice[2];
      Tymin = Tx + nx * latsum.RLattice[3] + ny * latsum.RLattice[4] + nz * latsum.RLattice[5];
      Tzmin = Tz + nx * latsum.RLattice[6] + ny * latsum.RLattice[7] + nz * latsum.RLattice[8];
    }
  }

  Tx = Txmin; Ty = Tymin; Tz = Tzmin;
}

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

void Add2(double * pOut, double const  pIn, double f, size_t n)
{
  size_t i = 0;
  for ( ; i < (n & ~3); i += 4 ) {
    pOut[i]   += f * pIn;
    pOut[i+1] += f * pIn;
    pOut[i+2] += f * pIn;
    pOut[i+3] += f * pIn;
  }
  pOut += i;
  switch(n - i) {
    case 3: pOut[2] += f*pIn;
    case 2: pOut[1] += f*pIn;
    case 1: pOut[0] += f*pIn;
    default: break;
  }
}


LatticeSum::LatticeSum(double* Lattice, double _Eta2Rho) {
  
  int nr = 1, nk = 10;

  int ir = 0; int Nr = 2*nr+1;
  vector<double> Rcoordcopy(3*Nr*Nr*Nr), Rdistcopy(Nr*Nr*Nr);
  Rcoord.resize(3 * Nr*Nr*Nr);
  Rdist.resize(Nr*Nr*Nr);

  //rvals
  for (int i = -nr; i<=nr ; i++)
  for (int j = -nr; j<=nr ; j++)
  for (int k = -nr; k<=nr ; k++) {
    Rcoordcopy[3*ir+0] = i * Lattice[0] + j * Lattice[3] + k * Lattice[6];
    Rcoordcopy[3*ir+1] = i * Lattice[1] + j * Lattice[4] + k * Lattice[7];
    Rcoordcopy[3*ir+2] = i * Lattice[2] + j * Lattice[5] + k * Lattice[8];

    Rdistcopy[ir] = Rcoordcopy[3*ir+0] * Rcoordcopy[3*ir+0]
        +  Rcoordcopy[3*ir+1] * Rcoordcopy[3*ir+1]
        +  Rcoordcopy[3*ir+2] * Rcoordcopy[3*ir+2];
    ir++;
  }
  
  //sort rcoord and rdist in ascending order
  std::vector<int> idx(Nr*Nr*Nr);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&Rdistcopy](size_t i1, size_t i2) {return Rdistcopy[i1] < Rdistcopy[i2];});
  for (int i=0; i<idx.size(); i++) {
    Rdist[i] = Rdistcopy[idx[i]];
    Rcoord[3*i+0] = Rcoordcopy[3*idx[i]+0];
    Rcoord[3*i+1] = Rcoordcopy[3*idx[i]+1];
    Rcoord[3*i+2] = Rcoordcopy[3*idx[i]+2];
  }

  //make klattice
  KLattice.resize(9), RLattice.resize(9);
  for (int i=0; i<9; i++) RLattice[i] = Lattice[i];
  RVolume = getKLattice(&KLattice[0], Lattice);

  int Nk = 2*nk +1;
  vector<double> Kcoordcopy(3*(Nk*Nk*Nk-1)), Kdistcopy(Nk*Nk*Nk-1);
  Kcoord.resize(3 * (Nk*Nk*Nk-1));
  Kdist.resize(Nk*Nk*Nk-1);

  ir = 0;
  //kvals
  for (int i = -nk; i<=nk ; i++)
  for (int j = -nk; j<=nk ; j++)
  for (int k = -nk; k<=nk ; k++) {
    if (i == 0 && j == 0 && k == 0) continue;
    Kcoordcopy[3*ir+0] = i * KLattice[0] + j * KLattice[3] + k * KLattice[6];
    Kcoordcopy[3*ir+1] = i * KLattice[1] + j * KLattice[4] + k * KLattice[7];
    Kcoordcopy[3*ir+2] = i * KLattice[2] + j * KLattice[5] + k * KLattice[8];

    Kdistcopy[ir] = Kcoordcopy[3*ir+0] * Kcoordcopy[3*ir+0]
                 +  Kcoordcopy[3*ir+1] * Kcoordcopy[3*ir+1]
                 +  Kcoordcopy[3*ir+2] * Kcoordcopy[3*ir+2];

    ir++;
  }

  idx.resize(Nk*Nk*Nk-1);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&Kdistcopy](size_t i1, size_t i2) {return Kdistcopy[i1] < Kdistcopy[i2];});
  
  for (int i=0; i<idx.size(); i++) {
    Kdist[i] = Kdistcopy[idx[i]];
    Kcoord[3*i+0] = Kcoordcopy[3*idx[i]+0];
    Kcoord[3*i+1] = Kcoordcopy[3*idx[i]+1];
    Kcoord[3*i+2] = Kcoordcopy[3*idx[i]+2];
  }


  Eta2Rho = _Eta2Rho / Rdist[1] ;

  //cout << omega<<"  "<<_Eta2Rho<<endl;
  //one can check the error if the real summation is truncated at n=1 (this is conservative)
  //with the given Eta2Rho
  /*
  double boys, boyseta;
  IrBoysFn(&boys, 2.* Rdist[1] * (Eta2Rho+1.), 0, 1.);
  IrBoysFn(&boyseta, 2 * Rdist[1] * Eta2Rho, 0, 1.);
  double error = boys - sqrt(Eta2Rho/(Eta2Rho+1)) * boyseta;
  cout << Eta2Rho<<"  "<<_Eta2Rho<<"  "<<Rdist[1]<<"  "<<error<<endl;
  */
}

void getAttenuatedBoys(double* pFmT, double T, int L, double factor,
                       double eta, ct::FMemoryStack& Mem) {
  IrBoysFn(pFmT, T, L, factor);

  double* pFmtOmega;
  Mem.Alloc(pFmtOmega, L+1);
  IrBoysFn(pFmtOmega, eta*eta*T, L, factor);

  double etaPow = eta;
  for (int i=0; i<L+1; i++) {
    pFmT[i] -= pFmtOmega[i] * etaPow;
    etaPow *= eta * eta;
  }
  
  Mem.Free(pFmtOmega);
}

// output: contracted kernels Fm(rho,T), format: (TotalL+1) x nCoA x nCoC
void Int2e2c_EvalCoKernels(double *pCoFmT, uint TotalL,
                           BasisShell *pA, BasisShell *pC,
                           double Tx, double Ty, double Tz,
                           double PrefactorExt, double* pInv2Alpha, double* pInv2Gamma,
                           LatticeSum& latsum, int rindex, ct::FMemoryStack &Mem)
{
  double t = Tx*Tx + Ty*Ty + Tz*Tz;
  double *pFmT;
  Mem.Alloc(pFmT, TotalL + 1); // FmT for current primitive.
  
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
    if (Rho <= latsum.Eta2Rho) continue;
    else {
      eta = sqrt(latsum.Eta2Rho/Rho);
    }

    //cout << eta<<"  "<<eta * sqrt(Rho/(1. - eta*eta))<<endl;
    // calculate derivatives (D/Dt)^m exp(-rho t) with t = (A-C)^2.
    getAttenuatedBoys(pFmT, Rho*t, TotalL, Prefactor*(2*M_PI)/Rho, eta, Mem);
    
    
    // convert from Gm(rho,T) to Fm(rho,T) by absorbing powers of rho
    // (those would normally be present in the R of the MDRR)
    double
        RhoPow = 1.;
    for ( uint i = 0; i < TotalL + 1; ++ i ){
      pFmT[i] *= RhoPow;
      RhoPow *= 2*Rho;
    }
    
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


void makeRealSummation(double *&pOutR, unsigned &TotalCo, BasisShell *pA, BasisShell *pC, double Tx, double Ty, double Tz, double Prefactor,   unsigned TotalLab, double* pInv2Alpha, double* pInv2Gamma, LatticeSum& latsum, ct::FMemoryStack& Mem) {
  double* pOutR_loc, *pCoFmT;
  int L = TotalLab;

  Mem.Alloc(pOutR_loc, (L+1)*(L+2)/2 * TotalCo);


  for (int r = 0; r<latsum.Rdist.size(); r++) {
    double Tx_r = Tx + latsum.Rcoord[3*r],
        Ty_r =  Ty + latsum.Rcoord[3*r+1],
        Tz_r =  Tz + latsum.Rcoord[3*r+2];

    Mem.ClearAlloc(pCoFmT, (L+1) * TotalCo);
    Int2e2c_EvalCoKernels(pCoFmT, L, pA, pC, Tx_r, Ty_r, Tz_r, Prefactor, pInv2Alpha, pInv2Gamma, latsum, r, Mem);
    
    //go from [0]^m -> [r]^0 using mcmurchie-davidson
    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
    for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {

      double
          *pFmT = &pCoFmT[(L+1)*(iCoA + pA->nCo*iCoC)],
          *pDataR_ = &pOutR_loc[nCartY(TotalLab) * (iCoA + pA->nCo*iCoC)];
    
      ShellMdrr(pDataR_, pFmT, Tx_r, Ty_r, Tz_r, L);
    }

    for (int i=0; i<(L+1)*(L+2)/2 * TotalCo; i++)
      pOutR[i] += pOutR_loc[i];
  }
}

void makeReciprocalSummation(double *&pOutR, unsigned &TotalCo, BasisShell *pA, BasisShell *pC, double Tx, double Ty, double Tz, double Prefactor,   unsigned TotalLab, double* pInv2Alpha, double* pInv2Gamma, LatticeSum& latsum, ct::FMemoryStack& Mem)
{
  int L = TotalLab;
  int LA = pA->l, LC = pC->l;

  //calcualte the reciprocal space contribution
  double background ;   
  double *pReciprocalSum;
  Mem.Alloc(pReciprocalSum, (L+1)*(L+1)/2);
  
  //double Lx = 1.0, Ly = 1.0, Lz = 1.0;
  //now go back to calculating contribution from reciprocal space
  for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
  for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
  {
    double
        Alpha = pA->exponents[iExpA],
        Gamma = pC->exponents[iExpC],
        InvEta = 1./(Alpha + Gamma),
        Rho = (Alpha * Gamma)*InvEta; // = (Alpha * Gamma)*/(Alpha + Gamma)
    
    
    double Eta = Rho <= latsum.Eta2Rho ? 1. : sqrt(latsum.Eta2Rho/Rho);
    double Omega = Rho <= latsum.Eta2Rho ? 1.0e9 : sqrt(Rho * Eta * Eta /(1. - Eta * Eta)) ; 
    double Eta2rho = Eta * Eta * Rho;


    double scale = Prefactor *
        M_PI*M_PI*M_PI*M_PI/pow(Alpha*Gamma, 1.5)/ latsum.RVolume; 
    scale *= pInv2Gamma[iExpC] * pInv2Alpha[iExpA];


    for (int i=0; i<(L+1)*(L+2)/2; i++)
      pReciprocalSum[i] = 0.0;


    for (int k=0; k<latsum.Kdist.size(); k++) {
      double expVal = exp(-latsum.Kdist[k]/Eta2rho/4.)/(latsum.Kdist[k]/4.);
      
      double maxG = getHermiteReciprocal(LA+LC, pReciprocalSum,
                                         latsum.Kcoord[3*k+0],
                                         latsum.Kcoord[3*k+1], latsum.Kcoord[3*k+2],
                                         Tx, Ty, Tz,
                                         expVal, scale);
      if (abs(maxG * scale * expVal) < 1.e-12) break;
    }      
    
    
    if (!(Rho <= latsum.Eta2Rho) && LA == 0 && LC == 0) {
      for (int i=0; i<(L+1)*(L+2)/2; i++)
        pReciprocalSum[i] -= M_PI * 16.*M_PI*M_PI * tgamma((LA+3)/2.)/(2. * pow(Alpha,0.5*(LA+3)))
            * tgamma((LC+3)/2.)/(2. * pow(Gamma,0.5*(LC+3))) * latsum.RVolume/Omega/Omega;
    }


    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
    for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
      double CoAC = pC->contractions(iExpC, iCoC) *
          pA->contractions(iExpA, iCoA);

      Add2(&pOutR[(L+1)*(L+2)/2*(iCoA + pA->nCo*iCoC)],
           pReciprocalSum, CoAC, (L+1)*(L+2)/2);
    }
  }
  
  Mem.Free(pReciprocalSum);
}


void Int2e2c_EvalCoShY(double *&pOutR, unsigned &TotalCo, BasisShell *pA, BasisShell *pC, double Prefactor,   unsigned TotalLab, double* pInv2Alpha, double* pInv2Gamma, LatticeSum& latsum, ct::FMemoryStack& Mem)
{
  //CHANGE THIS to minimum distance between A and periodic image of C
  double Tx, Ty, Tz;
  getRelativeCoords(pA, pC, latsum, Tx, Ty, Tz);
  
  TotalCo = pA->nCo * pC->nCo;
  
  unsigned
      L = TotalLab ;

  Mem.ClearAlloc(pOutR, (L+1)*(L+2)/2 * TotalCo);

  makeRealSummation(pOutR, TotalCo, pA, pC, Tx, Ty, Tz, Prefactor, TotalLab, pInv2Alpha, pInv2Gamma, latsum, Mem);
  makeReciprocalSummation(pOutR, TotalCo, pA, pC, Tx, Ty, Tz, Prefactor, TotalLab, pInv2Alpha, pInv2Gamma, latsum, Mem);



  
}

void EvalInt2e2c( double *pOut, size_t StrideA, size_t StrideC,
                  BasisShell *pA, BasisShell *pC, double Prefactor, bool Add,
                  LatticeSum& latsum, ct::FMemoryStack &Mem )
{
   uint
      lc = pC->l, la = pA->l,
      TotalCo;
   double
      *pDataR, *pR1, *pFinal;


   //we will need these in both the real and reciprocal space summations
   double
       *pInv2Alpha, *pInv2Gamma;
   Mem.Alloc(pInv2Alpha, pA->nFn);
   Mem.Alloc(pInv2Gamma, pC->nFn);  
   for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
     pInv2Alpha[iExpA] = bool(pA->l)? std::pow(+1.0/(2*pA->exponents[iExpA]), (int)pA->l) : 1.;
   for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
     pInv2Gamma[iExpC] = bool(pC->l)? std::pow(-1.0/(2*pC->exponents[iExpC]), (int)pC->l) : 1.;
  
   
   Int2e2c_EvalCoShY(pDataR, TotalCo, pA, pC, Prefactor, la + lc, pInv2Alpha, pInv2Gamma, latsum, Mem);
   
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
