#include <iostream>
#include <fstream>
#include <cmath>
#include <string.h>
#include <algorithm>
#include <numeric> 
#include <complex>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "GeneratePolynomials.h"
#include "IrAmrr.h"
#include "IrBoysFn.h"
#include "BasisShell.h"
#include "LatticeSum.h"
#include "interface.h"
#include "Integral3c_Boys.h"
#include "CxAlgebra.h"
#include "timer.h"

using namespace std;
using namespace std::chrono;
using namespace ir;

static size_t *MakeFnOffsets(const BasisShell* pCs, size_t nC, ct::FMemoryStack2 &Mem)
{
   size_t *piFnC;
   Mem.Alloc(piFnC, nC+1);
   Mem.Align(16);
   piFnC[0] = 0;
   for (size_t iC = 0; iC < nC; ++ iC)
      piFnC[iC+1] = piFnC[iC] + pCs[iC].numFuns();
   return piFnC;
}

struct FGaussProduct
{
   double
      ExpA, ExpB, // exponents of the primitives
      Eta, // ExpA + ExpB
      InvEta, // 1/(ExpA+ExpB)
      Exp, // exponent of the primitive product.
      DistSq; // squared distance between A and B
   double
      vCen[4],
      vAmB[4];
  FGaussProduct(double XA, double YA, double ZA, double ExpA_,
                double XB, double YB, double ZB, double ExpB_);

   double Sab() {
      return std::exp(-Exp * DistSq);
   }
};


inline void SubVec3(double *pOut, double const *pA, double const *pB) {
   pOut[0] = pA[0] - pB[0];
   pOut[1] = pA[1] - pB[1];
   pOut[2] = pA[2] - pB[2];
}
static double sqr(double x) { return x*x; }

// return x^(3/2).
inline double pow15(double x) {
   return x * std::sqrt(x);
}

inline double DistSq3(double const *pA, double const *pB) {
   return sqr(pA[0] - pB[0]) + sqr(pA[1] - pB[1]) + sqr(pA[2] - pB[2]);
}

static void Contract1(double *pOut, double *pIn, size_t nSize, const BasisShell *pC, uint iExpC)
{
  DGER(nSize, pC->nCo, 1.0, pIn, 1, &(pC->contractions(iExpC, 0)), pC->nFn, pOut, nSize);
}


FGaussProduct::FGaussProduct(double XA, double YA, double ZA, double ExpA_,
                             double XB, double YB, double ZB, double ExpB_)
{
   ExpA = ExpA_;
   ExpB = ExpB_;
   Eta = ExpA + ExpB;
   InvEta = 1./Eta;
   Exp = ExpA * ExpB * InvEta;

   double vA0 = XA, vA1 = YA, vA2 = ZA;
   double vB0 = XB, vB1 = YB, vB2 = ZB;

   vCen[0] = InvEta * (ExpA * XA + ExpB * XB);
   vCen[1] = InvEta * (ExpA * YA + ExpB * YB);
   vCen[2] = InvEta * (ExpA * ZA + ExpB * ZB);
   vAmB[0] = XA - XB;
   vAmB[1] = YA - YB;
   vAmB[2] = ZA - ZB;
   DistSq = sqr(vAmB[0]) + sqr(vAmB[1]) + sqr(vAmB[2]);
}


void EvalInt2e3c(double *pOut, size_t *Strides,
                 BasisShell *pA, BasisShell *pB, BasisShell *pCs,
                 size_t nC, double Prefactor, Kernel *pKernel,
                 LatticeSum& latsum, ct::FMemoryStack2 &Mem)
{
  double Eta2Rho = latsum.Eta2RhoCoul;
  void
      *pBaseOfMemory = Mem.Alloc(0);
   size_t
      StrideA = Strides[0], StrideB = Strides[1], StrideC = Strides[2];

   if (pA->l < pB->l) { // <- OsrrC only implemented for la >= lb.
      std::swap(pA, pB);
      std::swap(StrideA, StrideB);
   }
   double dbl = 0.0;
   
   // count number of C functions and find largest lc for memory purposes.
   size_t
      *piFnC = MakeFnOffsets(pCs, nC, Mem),
      nFnC_Total = piFnC[nC];
   uint
      lc_Max = 0;
   for (size_t iC = 0; iC < nC; ++ iC)
      lc_Max = std::max(lc_Max, (uint)pCs[iC].l);

   // allocate intermediates
   size_t
      nCartX_AB = nCartX(pA->l + pB->l),
      nCartX_Am1 = nCartX(pA->l-1),
      nCartX_B = nCartX(pB->l),
      nCartX_ABmA = nCartX_AB - nCartX_Am1,
      nShA_CartXB = pA->nSh() * nCartX_B;
   // intermediates for primitive integrals
   double
      // Kernel derivatives (-d/dT)^m of (00|00) integral
      *pFmT = Mem.AllocN(pA->l + pB->l + lc_Max + 1, dbl),
      // (a0|0) intermediate
      *p_A00 = Mem.AllocN(nCartX_AB, dbl),
      *p_A0C_sh_mem = Mem.AllocN(nCartX_ABmA * (2*lc_Max+1), dbl),
      *pMemOsrrB = Mem.AllocN(nCartX_AB * nCartX(lc_Max), dbl);
   // intermediates for contractions
   double
      // intermediates (a0|c) with AB primitives and C contracted, a = la..lab
      *p_A0C_ppc = Mem.AllocN(nCartX_ABmA * nFnC_Total, dbl),
      // intermediates (a0|c) with A,C contracted, a = la..lab.
      *p_A0C_cpc = Mem.AllocN(nCartX_ABmA * nFnC_Total * pA->nCo, dbl),
      // intermediates (a0|c) with A,B,C all contracted, a = la..lab.
      *p_A0C_ccc = Mem.ClearAllocN(nCartX_ABmA * nFnC_Total * pA->nCo * pB->nCo, dbl),
      // intermediates (xa|c) with A,B,C contracted and (xa| = nCartX(lb) x (2*la+1)
      *p_xAC_ccc = Mem.AllocN(nShA_CartXB * nFnC_Total * pA->nCo * pB->nCo, dbl);

   double Bx =  pB->Xcoord, By =  pB->Ycoord, Bz =  pB->Zcoord;
   double Ax =  pA->Xcoord, Ay =  pA->Ycoord, Az =  pA->Zcoord;
   double Tx, Ty, Tz;
   latsum.getRelativeCoords(pA, pB, Tx, Ty, Tz);
   double Ax0 = Bx+Tx, Ay0 = By+Ty, Az0 = Bz+Tz;
   //double Ax0 = Ax, Ay0 = Ay, Az0 = Az;

   int T1 = latsum.indexCenter(*pA);
   int T2 = latsum.indexCenter(*pB);
   int T = T1 == T2 ? 0 : max(T1, T2)*(max(T1, T2)+1)/2 + min(T1, T2);
   vector<size_t>& Tidx = latsum.ROrderedIdx[T];

   double screen = latsum.Rscreen;
   double logscreen = log(screen);
   vector<double>& KLattice = latsum.KLattice;
   vector<double>& RLattice = latsum.RLattice;

   int nQ = latsum.Rdist.size();
   for (int Q = 0; Q<nQ; Q++) {
     double Ax = Ax0 + latsum.Rcoord[3*Tidx[Q]+0];
     double Ay = Ay0 + latsum.Rcoord[3*Tidx[Q]+1];
     double Az = Az0 + latsum.Rcoord[3*Tidx[Q]+2];
     Tx = Ax-Bx; Ty = Ay-By; Tz = Az-Bz;

     bool FoundNonZero = false;
     
     memset(p_A0C_ccc, 0, nCartX_ABmA * nFnC_Total *  pA->nCo * pB->nCo * sizeof(*p_A0C_ccc));
     for (uint iExpB = 0; iExpB < pB->nFn; ++ iExpB) {
       memset(p_A0C_cpc, 0, nCartX_ABmA * nFnC_Total * pA->nCo * sizeof(*p_A0C_cpc));

       
       for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA) {
         
         double fDistSqAB = Tx*Tx+Ty*Ty+Tz*Tz;
         
         FGaussProduct
             OvAB(Ax, Ay, Az, pA->exponents[iExpA],
                  Bx, By, Bz, pB->exponents[iExpB]);

         if (OvAB.Eta <= Eta2Rho) continue;
         if (-OvAB.Exp*fDistSqAB < logscreen) continue;
         

         //find the nearest grid point
         double ABx0 = OvAB.vCen[0], ABy0 = OvAB.vCen[1], ABz0 = OvAB.vCen[2];
         int abx0= std::round((KLattice[0]*ABx0 + KLattice[1]*ABy0 + KLattice[2]*ABz0)/2/M_PI);
         int aby0= std::round((KLattice[3]*ABx0 + KLattice[4]*ABy0 + KLattice[5]*ABz0)/2/M_PI);
         int abz0= std::round((KLattice[6]*ABx0 + KLattice[7]*ABy0 + KLattice[8]*ABz0)/2/M_PI);
         ABx0 = abx0*RLattice[0]+aby0*RLattice[3]+abz0*RLattice[6];
         ABy0 = abx0*RLattice[1]+aby0*RLattice[4]+abz0*RLattice[7];
         ABz0 = abx0*RLattice[2]+aby0*RLattice[5]+abz0*RLattice[8];
         
         // ^- P == OvAB.vCen
         double
             Sab = std::exp(-OvAB.Exp * fDistSqAB), // [1] (6)
             PmA[3];
         PmA[0] = OvAB.vCen[0] - Ax;
         PmA[1] = OvAB.vCen[1] - Ay;
         PmA[2] = OvAB.vCen[2] - Az;
         
         memset(p_A0C_ppc, 0, nCartX_ABmA * nFnC_Total * sizeof(*p_A0C_ppc));
         for (size_t iC = 0; iC < nC; ++ iC) {
           BasisShell  *pC = &pCs[iC];
           double Cx0 = pC->Xcoord, Cy0=pC->Ycoord, Cz0 = pC->Zcoord;
           //double Cx = pC->Xcoord, Cy=pC->Ycoord, Cz = pC->Zcoord;
           
           for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
           {
             uint
                 TotalL = pA->l + pB->l + pC->l;
             double c = pC->exponents[iExpC];
             if (OvAB.Eta*c/(OvAB.Eta+c) <=Eta2Rho) continue;
             
             int nP = 27;//latsum.Rdist.size();//0 +/- x,y,z
             for (int P = 0; P<nP; P++) {
               double Cx = Cx0+ABx0+latsum.Rcoord[3*P+0];
               double Cy = Cy0+ABy0+latsum.Rcoord[3*P+1];
               double Cz = Cz0+ABz0+latsum.Rcoord[3*P+2];

               FGaussProduct
                   OvPC(OvAB.vCen[0], OvAB.vCen[1], OvAB.vCen[2], OvAB.Eta,
                        Cx, Cy, Cz, pC->exponents[iExpC]);
               double
                   *PmC = OvPC.vAmB,
                   Rho = OvPC.Exp, // [1] (3)
                   t = OvPC.DistSq,
                   Factor = pow15(M_PI * OvPC.InvEta) * Sab * Prefactor; // [1] (7)

               double eta = 1.0;
               if (Eta2Rho*t < 29) {
                 FoundNonZero = true;
                 eta = sqrt(Eta2Rho/Rho);
                 
                 pKernel->getValueRSpace(pFmT, Rho*t, TotalL, Factor, Rho, eta, Mem);

                 // make (a0|0)^m for a = 0..lab with lab + la+lb.
                 OsrrA(p_A00, pFmT + pC->l, (pA->l + pB->l), PmA[0], PmA[1], PmA[2],
                       PmC[0], PmC[1], PmC[2], Rho, OvAB.InvEta);
                 
                 // make (a0|c) for a = la..lab, c = 0..lc.
                 double
                     *p_A0C_sh;
                 if (pC->l == 0) {
                   p_A0C_sh = p_A00 + nCartX_Am1;
                 } else {
                   p_A0C_sh = p_A0C_sh_mem;
                   OsrrB_3c_shc(p_A0C_sh, p_A00, pMemOsrrB, pA->l, (pA->l + pB->l), pC->l,
                                PmC[0], PmC[1], PmC[2], OvPC.InvEta, Rho/pC->exponents[iExpC]);
                 }

                 Contract1(&p_A0C_ppc[nCartX_ABmA * piFnC[iC]], p_A0C_sh,
                           nCartX_ABmA*pC->nSh(), pC, iExpC);
               }
             }// P lattice
           } // c exponents
         } // c shells
         // p_A0C_ppc should be done now. Contract A and B.

         if (!FoundNonZero) continue;
         Contract1(p_A0C_cpc, p_A0C_ppc, nCartX_ABmA * nFnC_Total, pA, iExpA);
       } // a exponents

       
       if (!FoundNonZero) continue;
       Contract1(p_A0C_ccc, p_A0C_cpc, (nCartX_ABmA * nFnC_Total * pA->nCo), pB, iExpB);
     } // b exponents
     
     if (!FoundNonZero) continue;
     // transform A to solid harmonics by factoring nCartX(lab) into nCartX(lb) x Slm(A).
     ShTrA_XY(p_xAC_ccc, p_A0C_ccc, pA->l, (pA->l + pB->l), nFnC_Total * pA->nCo * pB->nCo);

     // we now have nCartX(lb) x nShA x nFnC_Total x nCoA x nCoB at p_xAC_ccc.
     // we still need to move the angular momentum from a to b and to write the
     // output integrals to their final destination.
     for (uint iCoB = 0; iCoB < pB->nCo; ++ iCoB)
       for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
         for (uint iFnC = 0; iFnC < nFnC_Total; ++ iFnC) {
           uint
               iFnA = iCoA * pA->nSh(), iFnB = iCoB * pB->nSh();
           OsrrCAppend(
               &pOut[iFnA*StrideA + iFnB*StrideB + iFnC*StrideC], StrideA, StrideB,
               &p_xAC_ccc[nShA_CartXB * (iFnC + (nFnC_Total * (iCoA + pA->nCo * iCoB)))],
               Tx, Ty, Tz, pB->l, pA->nSh() );
         };
       }
     
   } //Q lattice

   Mem.Free(pBaseOfMemory);
}




void PopulateAuxGMatrix(double* pOut, BasisShell* pC, size_t offset,
			size_t nAuxBas, LatticeSum& latsum, ct::FMemoryStack2 &Mem) {

  int lc = pC->l; int nterms = 2*lc +1;
  
  double *pInv2C;
  Mem.Alloc(pInv2C, pC->nFn);  
  for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
    pInv2C[iExpC]=bool(lc) ? std::pow(1.0/(2*pC->exponents[iExpC]), (int)lc) : 1.;
  
  
  double* pSphc; Mem.Alloc(pSphc, 2*lc+1);
  double Cx =  pC->Xcoord,
    Cy =  pC->Ycoord,
    Cz =  pC->Zcoord;
  
  double screen = latsum.Kscreen;
  double Eta2Rho =  latsum.Eta2RhoCoul;
  
  double logscreen = log(screen);
  
  for (int g=1; g<latsum.Kdist.size(); g++) {
    double Gx=latsum.Kcoord[3*g+0],
      Gy=latsum.Kcoord[3*g+1],
      Gz=latsum.Kcoord[3*g+2];
    
    
    if (-(Gx*Gx+Gy*Gy+Gz*Gz)/4/Eta2Rho < logscreen) break;
    ir::EvalSlcX_Deriv0(pSphc, Gx, Gy, Gz, lc);
      
    for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC) {
	
      double c = pC->exponents[iExpC],
          prefactor = 1./pow(c, 1.5) * pInv2C[iExpC];
      
      double *cMat = &pSphc[lc*lc];

      ct::Add2(&pOut[iExpC*nterms + offset + g *nAuxBas], cMat, prefactor, nterms);
    }
  }
  Mem.Free(pInv2C);

}

void PopulateAuxGMatrix(double* pOut, BasisSet& basis, std::vector<int>& shls,
			LatticeSum& latsum, ct::FMemoryStack2 &Mem) {

  size_t nAuxbas = basis.getNPrimitivebas(shls[5]) - basis.getNPrimitivebas(shls[4]);
  int nFnC = 0;
  for (int shlc = shls[4]; shlc < shls[5]; shlc++) {
    BasisShell *pC = &basis.BasisShells[shlc];
    PopulateAuxGMatrix(pOut, pC, nFnC, nAuxbas, latsum, Mem);

    nFnC += (2*pC->l+1) * pC->nFn; 
  }
}


void PopulatePairGMatrixKspace(double* pOutCos, double* pOutSin,
			       BasisShell* pA, BasisShell* pB,
			       int iExpA, int iExpB,
			       LatticeSum& latsum, ct::FMemoryStack2 &Mem) {

  pairKTime.start();
  
  int la = pA->l, lb = pB->l; int nterms = (2*la +1)*(2*lb+1);
  double* pReciprocalSumCos; Mem.Alloc(pReciprocalSumCos, nterms);
  double* pReciprocalSumSin; Mem.Alloc(pReciprocalSumSin, nterms);

  double pInv2A =  bool(la)? std::pow(1.0/(2*pA->exponents[iExpA]), (int)la) : 1.;
  double pInv2B =  bool(lb)? std::pow(1.0/(2*pB->exponents[iExpB]), (int)lb) : 1.;
  
  double* pSpha1; Mem.Alloc(pSpha1, (la+1)*(la+1));
  double* pSpha2; Mem.Alloc(pSpha2, (la+1)*(la+1));
  double* pSphb1; Mem.Alloc(pSphb1, (lb+1)*(lb+1));
  double* pSphb2; Mem.Alloc(pSphb2, (lb+1)*(lb+1));
  double Ax =  pA->Xcoord, Ay =  pA->Ycoord, Az =  pA->Zcoord;
  double Bx =  pB->Xcoord, By =  pB->Ycoord, Bz =  pB->Zcoord;

  
  double screen = latsum.Kscreen;
  double Eta2Rho =  latsum.Eta2RhoCoul;
  double PI11over2 = pow(M_PI, 5.5);
  
  vector<double>& KLattice = latsum.KLattice;
  vector<double>& RLattice = latsum.RLattice;
  
  double maxContraction = 0;
  for (uint iCoB = 0; iCoB < pB->nCo; ++ iCoB)
  for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
    double CoAB = pA->contractions(iExpA, iCoA) *
      pB->contractions(iExpB, iCoB);
    if (fabs(maxContraction) < fabs(CoAB))
      maxContraction = fabs(CoAB);
  }

  double
    a = pA->exponents[iExpA],
    b = pB->exponents[iExpB],
    alpha = a * b /(a + b),
    prefactor = PI11over2 / pow(a*b, 1.5) / pow(latsum.RVolume, 2);    
  double scale = prefactor * pInv2A * pInv2B ;
    
  double logscreen = log(screen) -
    log(maxContraction * scale);
    
  for (int g=0; g<latsum.KdistHalf.size(); g++) {
    double Gx=latsum.KcoordHalf[3*g+0],
      Gy=latsum.KcoordHalf[3*g+1],
      Gz=latsum.KcoordHalf[3*g+2];
    double Gsq = latsum.KdistHalf[g];
    double expArgG = -(Gsq)/4/min(a+b, Eta2Rho);
    
    if (expArgG < logscreen) break;
    bool foundNonZero = false;


    //find the nearest grid point
    double Fx0 = -b/(a+b)*Gx, Fy0 = -b/(a+b)*Gy, Fz0 = -b/(a+b)*Gz;
    int fx0= std::round((RLattice[0]*Fx0 + RLattice[1]*Fy0 + RLattice[2]*Fz0)/2/M_PI);
    int fy0= std::round((RLattice[3]*Fx0 + RLattice[4]*Fy0 + RLattice[5]*Fz0)/2/M_PI);
    int fz0= std::round((RLattice[6]*Fx0 + RLattice[7]*Fy0 + RLattice[8]*Fz0)/2/M_PI);
    Fx0 = fx0*KLattice[0]+fy0*KLattice[3]+fz0*KLattice[6];
    Fy0 = fx0*KLattice[1]+fy0*KLattice[4]+fz0*KLattice[7];
    Fz0 = fx0*KLattice[2]+fy0*KLattice[5]+fz0*KLattice[8];

    for (int i=0; i<nterms; i++) {
      pReciprocalSumCos[i] = 0.0;
      pReciprocalSumSin[i] = 0.0;
    }

    for (int f=0; f<latsum.Kdist.size(); f++) {
      double Fx=Fx0+latsum.Kcoord[3*f+0],
	Fy=Fy0+latsum.Kcoord[3*f+1],
	Fz=Fz0+latsum.Kcoord[3*f+2];
      
      double Fsq = Fx*Fx + Fy*Fy + Fz*Fz;
      double expArgF = - (Gx*Fx+Gy*Fy+Gz*Fz)/2./a - b*Gsq/4./a/(a+b) - Fsq/4./alpha;
      if (expArgF+expArgG > logscreen) {
	foundNonZero = true;      
	double expval2 = exp(expArgF);
      
	getSphReciprocal3cos(la, lb, pReciprocalSumCos,
			     pSpha1, pSpha2, pSphb1, pSphb2,
			     Gx, Gy, Gz,
			     Fx, Fy, Fz, Ax, Ay, Az,
			     Bx, By, Bz, expval2, scale);
	
	getSphReciprocal3sin(la, lb, pReciprocalSumSin,
			     pSpha1, pSpha2, pSphb1, pSphb2,
			     Gx, Gy, Gz,
			     Fx, Fy, Fz, Ax, Ay, Az,
			     Bx, By, Bz, expval2, scale);
      }
      
    }
    if (!foundNonZero) break;
    ct::Add2(&pOutCos[g*nterms], pReciprocalSumCos, 1.0, nterms);
    ct::Add2(&pOutSin[g*nterms], pReciprocalSumSin, 1.0, nterms);
  }
  Mem.Free(pReciprocalSumCos);
  pairKTime.stop();
  
}


void PopulatePairGMatrixRspace(double* pOutCos, double* pOutSin,
			       BasisShell* pA, BasisShell* pB,
			       int iExpA, int iExpB,
			       LatticeSum& latsum, ct::FMemoryStack2 &Mem) {

  pairRTime.start();
    
  int la = pA->l, lb = pB->l; int nterms = (2*la +1)*(2*lb+1);
  double* pReciprocalSumCos; Mem.Alloc(pReciprocalSumCos, nterms);
  double* pReciprocalSumSin; Mem.Alloc(pReciprocalSumSin, nterms);

  double pInv2A =  bool(la)? std::pow(1.0/(2*pA->exponents[iExpA]), (int)la) : 1.;
  double pInv2B =  bool(lb)? std::pow(1.0/(2*pB->exponents[iExpB]), (int)lb) : 1.;
  
  double Ax =  pA->Xcoord, Ay =  pA->Ycoord, Az =  pA->Zcoord;
  double Bx =  pB->Xcoord, By =  pB->Ycoord, Bz =  pB->Zcoord;

  double Tx, Ty, Tz;
  latsum.getRelativeCoords(pA, pB, Tx, Ty, Tz);
  //Bx = 2*Bx+Tx-Ax; By = 2*By+Ty-Ay; Bz = 2*Bz+Tz-Az;
  Bx = Ax-Tx; By = Ay-Ty; Bz = Az-Tz;

  int T1 = latsum.indexCenter(*pA);
  int T2 = latsum.indexCenter(*pB);
  int T = T1 == T2 ? 0 : max(T1, T2)*(max(T1, T2)+1)/2 + min(T1, T2);
  vector<size_t>& Tidx = latsum.ROrderedIdx[T];
  
  double screen = latsum.Rscreen;
  double Eta2Rho =  latsum.Eta2RhoCoul;
  
  vector<double>& KLattice = latsum.KLattice;
  vector<double>& RLattice = latsum.RLattice;
  
  double maxContraction = 0;
  for (uint iCoB = 0; iCoB < pB->nCo; ++ iCoB)
  for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
    double CoAB = pA->contractions(iExpA, iCoA) *
      pB->contractions(iExpB, iCoB);
    if (fabs(maxContraction) < fabs(CoAB))
      maxContraction = fabs(CoAB);
  }

  double
    a = pA->exponents[iExpA],
    b = pB->exponents[iExpB],
    alpha = a * b /(a + b),
    prefactor = M_PI*M_PI*M_PI*M_PI / pow(a + b, 1.5) / latsum.RVolume;    
  double scale = prefactor * pInv2A * pInv2B ;
    
  double logscreen = log(screen/(maxContraction * scale));

  for (int g=0; g<latsum.KdistHalf.size(); g++) {
    double Gx=latsum.KcoordHalf[3*g+0],
      Gy=latsum.KcoordHalf[3*g+1],
      Gz=latsum.KcoordHalf[3*g+2];
    double Gsq = latsum.KdistHalf[g];
    double expArgG = -(Gsq)/4/min(a+b, Eta2Rho);
    
    if (expArgG < logscreen) break;
    bool foundNonZero = false;


    for (int i=0; i<nterms; i++) {
      pReciprocalSumCos[i] = 0.0;
      pReciprocalSumSin[i] = 0.0;
    }

    for (int q=0; q<latsum.Rdist.size(); q++) {
      double Qx = -latsum.Rcoord[3*Tidx[q]+0],
	Qy = -latsum.Rcoord[3*Tidx[q]+1],
	Qz = -latsum.Rcoord[3*Tidx[q]+2];
      
      double expArgQ = -alpha * (    (Ax-Bx-Qx)*(Ax-Bx-Qx)
				  +  (Ay-By-Qy)*(Ay-By-Qy)
				     +  (Az-Bz-Qz)*(Az-Bz-Qz));
      
      if (expArgQ+expArgG < logscreen) break;
      foundNonZero = true;
      
      double expval2 = exp(expArgQ);

      getSphRealRecursion(la, lb, pReciprocalSumCos,
			  pReciprocalSumSin,
			  Gx, Gy, Gz,
			  Qx, Qy, Qz, Ax, Ay, Az,
			  Bx, By, Bz, alpha, a, b, expval2,
			  scale, Mem);

    }
    ct::Add2(&pOutCos[g*nterms], pReciprocalSumCos, 1.0, nterms);
    ct::Add2(&pOutSin[g*nterms], pReciprocalSumSin, 1.0, nterms);
    if (!foundNonZero) break;
  }
  Mem.Free(pReciprocalSumCos);
  pairRTime.stop();
  
}


void contractCoulombKernel(double* pOut, double* OrbPairGMatrixcos,
			   double* OrbPairGMatrixsin, double* preMadeIntegrals,
			   LatticeSum& latsum, BasisShell* pA, BasisShell *pB,
			   int iExpA, int iExpB, BasisShell* pC, int coffset,
			   ct::FMemoryStack2 &Mem) {

  coulombContractTime.start();
  
  double maxConA = 0, maxConB=0, maxConC=0;
  for (uint iCoB = 0; iCoB < pB->nCo; ++ iCoB)
    if (fabs(maxConB) < fabs(pB->contractions(iExpB,iCoB)))
      maxConB = fabs(pB->contractions(iExpB,iCoB));
  for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA)
    if (fabs(maxConA) < fabs(pA->contractions(iExpA,iCoA)))
      maxConA = fabs(pA->contractions(iExpA,iCoA));
  double maxContraction = maxConA*maxConB;


  double Eta2Rho =  latsum.Eta2RhoCoul;
  double logscreen = log(latsum.Kscreen/maxContraction);
  int Pidx = 0;

  double a = pA->exponents[iExpA], b = pB->exponents[iExpB];
  int la = pA->l, lb = pB->l;
  int ntermsa = (2*la+1), ntermsb = (2*lb+1), ntermsab = ntermsa*ntermsb;
  int lc = pC->l; int ntermsc = 2*lc+1; int nterms = ntermsab * ntermsc;
  double Cx = pC->Xcoord, Cy = pC->Ycoord, Cz = pC->Zcoord;
  
  int bstride = ntermsa, cstride = ntermsab;
  int nG = latsum.KdistHalf.size();
  
  double *pSphc; Mem.Alloc(pSphc, (lc+1)*(lc+1));


  int atomT = latsum.indexCenter(*pC); //find the index of the atom

  double* cosCommon = (&latsum.CosKval3c[atomT][0])+lc*lc;
  double* sinCommon = (&latsum.SinKval3c[atomT][0])+lc*lc;
  char n = 'N', t = 'T'; double alpha = 1.0, beta = 0.0; int one = 1;

  
  double *pReciprocal; Mem.Alloc(pReciprocal, ntermsab*ntermsc);
  double *pReciprocalLargeRho; Mem.Alloc(pReciprocalLargeRho, ntermsab*ntermsc);
  double *cosC; Mem.Alloc(cosC, (2*lc+1)*nG);
  double *sinC; Mem.Alloc(sinC, (2*lc+1)*nG);
  double *gkernel; Mem.ClearAlloc(gkernel, nG);

  
  int nfnC = 0;
  for (int iExpC = 0; iExpC < pC->nFn; iExpC++) {

    maxConC = 0.;
    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
      if (fabs(maxConC) < fabs(pC->contractions(iExpC,iCoC)))
        maxConC = fabs(pC->contractions(iExpC,iCoC));
    
    double c = pC->exponents[iExpC];
    double Rho = (a+b)*c/(a+b+c);
    double rho = min(Rho, Eta2Rho);
    double prefactor = 2./pow(c, 1.5) * std::pow(1.0/(2*c), lc);

    double logscreenABC = logscreen - log(maxConC*prefactor);

    if (Rho < Eta2Rho) {

      int g = 1;
      for (; g<latsum.KdistHalf.size(); g++) {
        double Gsq = latsum.KdistHalf[g];
        double expArgG = -(Gsq)/4/rho;
        
        if (expArgG < logscreenABC) break;
        gkernel[g] = exp(expArgG)/(Gsq/4.);
      }

      int index2 = 0, index=0;
      for (int i=0; i<g; i++) {
        ct::Add2_0(cosC+index, cosCommon+index2, gkernel[i], ntermsc);
        index += ntermsc; index2 += 49;
      }

      DGEMM(n, t, ntermsab, ntermsc, g, prefactor, OrbPairGMatrixcos, ntermsab,
            cosC, ntermsc,  beta, pReciprocal, ntermsab );
      

      index = 0; index2 = 0;
      for (int i=0; i<g; i++) {
        ct::Add2_0(sinC+index, sinCommon+index2, gkernel[i], ntermsc);
        index += ntermsc; index2 += 49;
      }
      

      DGEMM(n, t, ntermsab, ntermsc, g, prefactor, OrbPairGMatrixsin, ntermsab,
            sinC, ntermsc,  alpha, pReciprocal, ntermsab );

    }
    else {
      int index0 = atomT*ntermsab*49+ntermsab*lc*lc;
      for (int i=0; i<ntermsc*ntermsab; i++)
        pReciprocal[i] = preMadeIntegrals[index0+i]*prefactor;
    }


    
    if (lc == 0 && Rho > Eta2Rho) {
      double Eta = sqrt(Eta2Rho/Rho);
      double Omega2 = Rho * Eta * Eta /(1. - Eta * Eta) ;
      for (int ab=0;ab<ntermsab; ab++)
        pReciprocal[ab] -= prefactor*OrbPairGMatrixcos[ab]/Omega2/2.;
    }

    for (int iCoC = 0; iCoC < pC->nCo; ++iCoC) {
      double CoC = pC->contractions(iExpC, iCoC);
      for (int c = 0; c<ntermsc; c++) 
	for (int b = 0; b<ntermsb; b++)
	  for (int a = 0; a<ntermsa; a++) {
	    pOut[ a + b * ntermsa + (iCoC*ntermsc+c) * ntermsab]
	      += pReciprocal[a + b*ntermsa + ntermsab*c] * CoC;
	  }
    }

    nfnC += ntermsc;
  }

  Mem.Free(pSphc);
  coulombContractTime.stop();

}

void premakeHighRhoIntegrals(double* preMadeIntegrals,  double* OrbPairGMatrixcos,
                             double* OrbPairGMatrixsin, BasisShell *pA,
                             BasisShell *pB, int iExpA, int iExpB, LatticeSum& latsum,
                             ct::FMemoryStack2& Mem, BasisSet& basis,
                             vector<int>& shls) {
  int la = pA->l; int ntermsa = (2*la+1); double a= pA->exponents[iExpA];
  int lb = pB->l; int ntermsb = (2*lb+1); double b= pB->exponents[iExpB];
  int natom = latsum.atomCenters.size()/3, ntermsab = ntermsa*ntermsb;
  
  int* foundL; Mem.ClearAlloc(foundL, 7*natom);
  int nG = latsum.KdistHalf.size();


  double* gkernel = &latsum.CoulKernelG[0];
  
  for (int shlc = shls[4]; shlc <shls[5]; shlc++) {
    BasisShell *pC = &basis.BasisShells[shlc];    
    int lc = pC->l; int ntermsc = (2*lc+1);
    double *cosC; Mem.Alloc(cosC, (2*lc+1)*nG);
    double *sinC; Mem.Alloc(sinC, (2*lc+1)*nG);

    int atomT = latsum.indexCenter(*pC); //find the index of the atom
    double* cosCommon = (&latsum.CosKval3cWithGkernel[atomT][0])+lc*lc;
    double* sinCommon = (&latsum.SinKval3cWithGkernel[atomT][0])+lc*lc;
    char n = 'N', t = 'T'; double alpha = 1.0, beta = 0.0; int one = 1;

    if (foundL[7*atomT + lc] == 1) continue;

    for (int iExpC = 0; iExpC < pC->nFn; iExpC++) {
      double c = pC->exponents[iExpC];
      double Rho = (a+b)*c/(a+b+c);

      if (Rho < latsum.Eta2RhoCoul) continue;
      double prefactor = 1.0;//2./pow(c, 1.5) * std::pow(1.0/(2*c), lc);

      double* pReciprocal = preMadeIntegrals
          + atomT* ntermsab*49
          + ntermsab*lc*lc;

      DGEMM(n, t, ntermsab, ntermsc, nG, prefactor, OrbPairGMatrixcos, ntermsab,
            cosCommon, 49,  beta, pReciprocal, ntermsab );

      DGEMM(n, t, ntermsab, ntermsc, nG, prefactor, OrbPairGMatrixsin, ntermsab,
            sinCommon, 49,  alpha, pReciprocal, ntermsab );
      
      foundL[7*atomT+lc] = 1;

      break;
    }
    
  }
  Mem.Free(foundL);

}

void ThreeCenterIntegrals(std::vector<int>& shls, BasisSet& basis, std::vector<double>& Lattice, ct::FMemoryStack2& Mem) {

  //10 along R and G directions
  //LatticeSum latsum(&Lattice[0], 10, 10, Mem, basis, 1., 2., 1.e-11, 1e-11);
  LatticeSum latsum(&Lattice[0], 3, 8, Mem, basis, 1., 30., 1.e-12, 1e-12, false, true);

  //LatticeSum latsum(&Lattice[0], 20, 20, Mem, basis, 1., 30., 1.e-11, 1e-11);
  cout <<"et2rho-Coul "<< latsum.Eta2RhoCoul<<"  "<<*latsum.CoulKernelG.rbegin()<<endl;
  cout <<"et2rho-Ovlp "<< latsum.Eta2RhoOvlp<<endl;

  //first calculate the auxbasis G array and store it
  size_t nAuxbas = basis.getNbas(shls[5]) - basis.getNbas(shls[4]);

  //nbasis
  size_t nbas = basis.getNbas(shls[1]) - basis.getNbas(shls[0]);

  cout <<nAuxbas<<"  "<<nbas<<endl;

  vector<double> Integral3c(nbas*nbas*nAuxbas,0.0);
  Int2e3cRK(&Integral3c[0], shls, basis, latsum, Mem);
  cout <<"after k "<< Integral3c[0]<<endl;
  Int2e3cRR(&Integral3c[0], shls, basis, latsum, Mem);
  cout <<"after r "<< Integral3c[0]<<endl;

  vector<double> test(nbas*(nbas+1)/2*nAuxbas);
  for (int k=0; k<nAuxbas; k++)
  for (int j=0; j<nbas; j++)
  for (int i=0; i<=j; i++) {
    test[k * nbas*(nbas+1)/2 + j*(j+1)/2+i] = Integral3c[j + i*nbas + k *nbas*nbas];
  }
  string name = "int2e3c";
  ofstream file(name.c_str(), ios::binary);
  file.write(reinterpret_cast<char*>(&test[0]), test.size()*sizeof(double));
  file.close();
}

void Int2e3cRK(double *pOut, vector<int>& shls, BasisSet& basis, LatticeSum& latsum,
	       ct::FMemoryStack2 &Mem) {

  int nG = latsum.Kcoord.size();

  size_t nAuxbas = basis.getNbas(shls[5]) - basis.getNbas(shls[4]);
  size_t nbas = basis.getNbas(shls[1]) - basis.getNbas(shls[0]);

  int maxAuxShell = 0;
  for (int shlc =shls[4]; shlc < shls[5]; shlc++) {
    int nfn = basis.BasisShells[shlc].numFuns();
    if (maxAuxShell < nfn)
      maxAuxShell = nfn;
  }
  
  auto start = high_resolution_clock::now();

  //loop over basis shells
  int aoffset = 0;
  for (int shla = shls[0]; shla < shls[1]; shla++) {
    BasisShell *pA = &basis.BasisShells[shla];
    int la = pA->l;
    int ntermsa = (2*la+1);
  
    int boffset = 0;
    for (int shlb = shls[2]; shlb <= shla; shlb++) {
      BasisShell *pB = &basis.BasisShells[shlb];    
      
      int lb = pB->l;
      int ntermsb = (2*lb+1), ntermsab = (2*la+1)*(2*lb+1);
      double* OrbPairGMatrixCos; Mem.Alloc(OrbPairGMatrixCos, nG*ntermsab);
      double* OrbPairGMatrixSin; Mem.Alloc(OrbPairGMatrixSin, nG*ntermsab);

      size_t worklen = maxAuxShell*(2*la+1)*(2*lb+1);
      double* KspaceSum; Mem.ClearAlloc(KspaceSum, worklen);
      double* RspaceSum; Mem.ClearAlloc(RspaceSum, worklen);

      //cout << worklen<<endl;
      //loop over individual basis functions
      for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA) 
      for (uint iExpB = 0; iExpB < pB->nFn; ++ iExpB) {
	double a = pA->exponents[iExpA], b = pB->exponents[iExpB];
	
        
	for (int g=0; g<nG*ntermsab; g++){
	  OrbPairGMatrixCos[g] = 0.0;
	  OrbPairGMatrixSin[g] = 0.0;
	}
	
	double alpha = a*b/(a+b);
	if (alpha < latsum.Eta2RhoOvlp) {
	  PopulatePairGMatrixKspace(OrbPairGMatrixCos, OrbPairGMatrixSin,
				    pA, pB, iExpA, iExpB, latsum, Mem);
	}
	else {
	  PopulatePairGMatrixRspace(OrbPairGMatrixCos, OrbPairGMatrixSin,
				    pA, pB, iExpA, iExpB, latsum, Mem);
	}

        double *preMadeIntegrals;
        int natom = latsum.atomCenters.size()/3;
        Mem.ClearAlloc(preMadeIntegrals , ntermsab*49*natom);
        premakeHighRhoIntegrals(preMadeIntegrals, OrbPairGMatrixCos, OrbPairGMatrixSin,
                                pA, pB, iExpA, iExpB, latsum, Mem, basis, shls);
        
	int coffset = 0;
	for (int shlc = shls[4]; shlc <shls[5]; shlc++) {

	  BasisShell *pC = & basis.BasisShells[shlc];
	  size_t worklen = pC->numFuns()*(2*la+1)*(2*lb+1);
	  for (int i=0; i<worklen; i++) KspaceSum[i] = 0.;
	  	  
	  contractCoulombKernel(KspaceSum, OrbPairGMatrixCos,
				OrbPairGMatrixSin, preMadeIntegrals,
				latsum, pA, pB, iExpA, iExpB,
				pC, coffset, Mem);

	  int bstride = nbas     , bstrideInter = pA->numFuns();
	  int cstride = nbas*nbas, cstrideInter = bstrideInter * (2*lb+1);

	  
	  double* Inter1;
	  Mem.ClearAlloc(Inter1, pA->numFuns() * (2*lb+1) * pC->numFuns());
	  
	  int ntermsbc = (2*lb+1)*pC->numFuns();
	  for (int bc = 0; bc<ntermsbc; bc++)
	  for (int iCoA = 0; iCoA < pA->nCo; iCoA++) {
	    double CoA = pA->contractions(iExpA, iCoA);
	    for (int a = 0; a<ntermsa; a++)
	      Inter1[iCoA*ntermsa+a + bc * bstrideInter] +=
		CoA * KspaceSum[a + bc * ntermsa];
	  }

	  //cout << pA->numFuns()*(2*lb+1)*pC->numFuns()<<"  Inter size"<<endl;
	  for (int iCoB = 0; iCoB < pB->nCo; iCoB++) {
	    double CoB = pB->contractions(iExpB, iCoB);
	    for (int b = 0; b<ntermsb ; b++) 
	      for (int ic = 0; ic < pC->numFuns(); ic++) {
		for (int ia = 0; ia < pA->numFuns(); ia++) {
		  pOut[ ia + aoffset
			+ (iCoB*ntermsb + b + boffset) * bstride
			+ (ic + coffset) * cstride]
		    += CoB * Inter1[ia + b * bstrideInter
				    + ic * cstrideInter];
		}
	      }
	  }

	  //cout << "inter "<<pA->contractions(0,0)<<"  "<<pA->contractions(1,0)<<endl;
	  //cout <<pA->contractions(iExpA,0)<<"  "<<pB->contractions(iExpB,0)<<" "<<iExpA<<"  "<<iExpB<<"  "<<shlc<<"  "<< pOut[3]<<endl;
	  Mem.Free(Inter1);

	  coffset += pC->numFuns();
	}
        Mem.Free(preMadeIntegrals);

      }

      Mem.Free(OrbPairGMatrixCos);
      boffset += pB->numFuns();
    }
    aoffset += pA->numFuns();

  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout <<"Auxbas populate "<< duration.count()/1e6<<endl;
  cout << pairRTime<<"  "<<pairKTime<<"  "<<coulombContractTime<<endl;

}


void Int2e3cRR(double *pIntFai, vector<int>& shls, BasisSet &basis, LatticeSum& latsum, ct::FMemoryStack2 &Mem)
{
  CoulombKernel IntKernel;

  auto start = high_resolution_clock::now();

  void
      *pBaseOfMemory = Mem.Alloc(0);
   size_t
       nAo1 = basis.getNbas(shls[1]) - basis.getNbas(shls[0]),
       nAo2 = basis.getNbas(shls[3]) - basis.getNbas(shls[2]),
       nFit = basis.getNbas(shls[5]) - basis.getNbas(shls[4]);


   //double
   //*pDF_NFi;
   //Mem.Alloc(pDF_NFi, nAo1 * nFit * nAo2);
   int iabas = 0, ibbas = 0, ifbas = 0;
   for ( size_t iShF = shls[4]; iShF != shls[5]; ++ iShF ){
      BasisShell &ShF = basis.BasisShells[iShF];
      size_t nFnF = ShF.numFuns();

      ibbas = 0;
      for ( size_t iShB = shls[2]; iShB < shls[3]; ++ iShB ){
         BasisShell &ShB = basis.BasisShells[iShB];
         size_t nFnB = ShB.numFuns();
                  
         iabas = 0;
         for ( size_t iShA = shls[0]; iShA < shls[1]; ++ iShA ) {
           BasisShell &ShA = basis.BasisShells[iShA];
           size_t nFnA = ShA.numFuns(),
               Strides[3] = {1, nFnA, nFnA * nFnB};
           if (iShB <= iShA) {
             
             double
                 *pIntData;
             Mem.ClearAlloc(pIntData, nFnA * nFnB * nFnF );
             
             EvalInt2e3c(pIntData, Strides, &ShA, &ShB, &ShF,1, 1.0, &IntKernel, latsum, Mem);
             
             for ( size_t iF = 0; iF < nFnF; ++ iF )
               for ( size_t iB = 0; iB < nFnB; ++ iB )
                 for ( size_t iA = 0; iA < nFnA; ++ iA ) {
                   double
                       f = pIntData[iA + nFnA * (iB + nFnB * iF)];
                   pIntFai[ (iabas+iA) + nAo1 * ( (ibbas+iB) + (ifbas+iF) * nAo2)] += f;
                 }
             //exit(0);
           }
           iabas += nFnA;
         }
         ibbas += nFnB;
      }
      ifbas += nFnF;
   }
   Mem.Free(pBaseOfMemory);

   auto stop = high_resolution_clock::now();
   auto duration = duration_cast<microseconds>(stop - start);
   cout <<"rspace "<< duration.count()/1e6<<endl;
}


