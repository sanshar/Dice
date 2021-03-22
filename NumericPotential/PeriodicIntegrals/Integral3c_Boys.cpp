#include <iostream>
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

/*
static bool IsWithinRange(BasisShell const *pA, BasisShell const *pB) {
   if (!pA->pRange || !pB->pRange)
      return true; // shells have no screening data.
   return ir::sqr(pA->MaxCoRange() + pB->MaxCoRange()) >= DistSq3(pA->vCen, pB->vCen);
}

static bool IsPrimitiveWithinRange(BasisShell const *pA, uint iExpA, BasisShell const *pB, uint iExpB, double fDistSqAB)
{
   if (!pA->pRange || !pB->pRange)
      return true; // shells have no screening data.
   return ir::sqr(pA->ExpRange(iExpA) + pB->ExpRange(iExpB)) >= fDistSqAB;
}

static bool IsContractionWithinRange(BasisShell const *pA, uint iCoA, BasisShell const *pB, uint iCoB, double fDistSqAB)
{
   if (!pA->pRange || !pB->pRange)
      return true; // shells have no screening data.
   return ir::sqr(pA->CoRange(iCoA) + pB->CoRange(iCoB)) >= fDistSqAB;
}

static bool IsPrimitiveWithinRange(BasisShell const *pA, uint iExpA, BasisShell const *pB, uint iExpB)
{
   return IsPrimitiveWithinRange(pA, iExpA, pB, iExpB, DistSq3(pA->vCen, pB->vCen));
}
*/

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
                 BasisShell const *pA, BasisShell const *pB, BasisShell const *pCs,
                 size_t nC, double Prefactor, Kernel *pKernel,
                 LatticeSum& latsum, ct::FMemoryStack2 &Mem)
{
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

   double Tx, Ty, Tz;
   //latsum.getRelativeCoords(pA, pC, Tx, Ty, Tz);
   Tx = pA->Xcoord - pB->Xcoord;
   Ty = pA->Ycoord - pB->Ycoord;
   Tz = pA->Zcoord - pB->Zcoord;

   //FVec3
   //vAmB = FVec3(pA->vCen) - FVec3(pB->vCen);
   double
//       fRangeKernel = sqr(pKernel->MaxRange()),
      fDistSqAB = Tx*Tx+Ty*Ty+Tz*Tz;

   for (uint iExpB = 0; iExpB < pB->nFn; ++ iExpB)
   {
      memset(p_A0C_cpc, 0, nCartX_ABmA * nFnC_Total * pA->nCo * sizeof(*p_A0C_cpc));

      for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
      {
         // skip if Dist(A,B) < Range(A) + Range(B)
        //if (!IsPrimitiveWithinRange(pA, iExpA, pB, iExpB, fDistSqAB))
        //continue;

         FGaussProduct
             OvAB(pA->Xcoord, pA->Ycoord, pA->Zcoord, pA->exponents[iExpA],
                  pB->Xcoord, pB->Ycoord, pB->Zcoord, pB->exponents[iExpB]);
            // ^- P == OvAB.vCen
         double
            Sab = std::exp(-OvAB.Exp * fDistSqAB), // [1] (6)
            PmA[3];
         PmA[0] = OvAB.vCen[0] - pA->Xcoord;
         PmA[1] = OvAB.vCen[1] - pA->Ycoord;
         PmA[2] = OvAB.vCen[2] - pA->Zcoord;
         //SubVec3(PmA, OvAB.vCen, pA->vCen);

         memset(p_A0C_ppc, 0, nCartX_ABmA * nFnC_Total * sizeof(*p_A0C_ppc));
         for (size_t iC = 0; iC < nC; ++ iC) {
            BasisShell const *pC = &pCs[iC];
            uint
               TotalL = pA->l + pB->l + pC->l;

            for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
            {
               FGaussProduct
                   OvPC(OvAB.vCen[0], OvAB.vCen[1], OvAB.vCen[2], OvAB.Eta,
                        pC->Xcoord, pC->Ycoord, pC->Zcoord, pC->exponents[iExpC]);
               double
                   *PmC = OvPC.vAmB,
                   Rho = OvPC.Exp, // [1] (3)
                   t = OvPC.DistSq,
                   Factor = pow15(M_PI * OvPC.InvEta) * Sab * Prefactor; // [1] (7)

               // make I[m] = (00|0)^m, m = 0..TotalL (inclusive)
               //pKernel->EvalGm(pGm, rho, T, TotalL, Factor);

               double eta = 0.0;
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

               // (a0|c) with solid harmonic c is ready now. Just need to add it to
               // its contractions.
               Contract1(&p_A0C_ppc[nCartX_ABmA * piFnC[iC]], p_A0C_sh,
                  nCartX_ABmA*pC->nSh(), pC, iExpC);
            } // c exponents
         } // c shells
         // p_A0C_ppc should be done now. Contract A and B.
         Contract1(p_A0C_cpc, p_A0C_ppc, nCartX_ABmA * nFnC_Total, pA, iExpA);
      } // a exponents
      Contract1(p_A0C_ccc, p_A0C_cpc, (nCartX_ABmA * nFnC_Total * pA->nCo), pB, iExpB);
   } // b exponents
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
            OsrrC(
               &pOut[iFnA*StrideA + iFnB*StrideB + iFnC*StrideC], StrideA, StrideB,
               &p_xAC_ccc[nShA_CartXB * (iFnC + (nFnC_Total * (iCoA + pA->nCo * iCoB)))],
               Tx, Ty, Tz, pB->l, pA->nSh() );
         };
      }

   Mem.Free(pBaseOfMemory);
}


//this will do contractions and will give out solidharmonics for all pairs of
//contracted gaussians in the two shells
//the total amount of memory in pout is nG x (2la+1) x (2lb+1) x nCoa x nCob
// assuming nG = 1000, la=6, lb = 6, nCoa=10, nCob=10, it will have 10^7 numbers
//requiring 0.1 Gb which should be very doable
/*
void EvalInt2e2cKarrayKsum(double *pOut, BasisShell const *pA,
			   BasisShell const *pB, double Prefactor, Kernel *pKernel,
			   LatticeSum& latsum, ct::FMemoryStack2 &Mem) {


  double
    *pInv2A, *pInv2B;
  Mem.Alloc(pInv2A, pA->nFn);
  Mem.Alloc(pInv2B, pB->nFn);  
  for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
    pInv2A[iExpA] = bool(pA->l)? std::pow(1.0/(2*pA->exponents[iExpA]), (int)pA->l) : 1.;
  for (uint iExpB = 0; iExpB < pB->nFn; ++ iExpB)
    pInv2B[iExpB] = bool(pB->l)? std::pow(1.0/(2*pB->exponents[iExpB]), (int)pB->l) : 1.;
  
  int la = pA->l, lb = pB->l;

  double TAx =  pA->Xcoord,
    TAy =  pA->Ycoord,
    TAz =  pA->Zcoord;

  double TABx =  pA->Xcoord - pB->Xcoord,
    TABy =  pA->Ycoord - pB->Ycoord,
    TABz =  pA->Zcoord - pB->Zcoord;

  int nterms = (2*la+1)*(2*lb+1);

  double *pReciprocalSum;
  Mem.Alloc(pReciprocalSum, nterms);
  
  double *pSpha, *pSphb, *pSphc;
  Mem.Alloc(pSpha, ((la+1)*(la+1)));
  Mem.Alloc(pSphb, ((lb+1)*(lb+1)));
  

  double screen = latsum.Kscreen;
  double PI11over2 = pow(M_PI, 5.5);
  
  ksumTime2.start();

  //now go back to calculating contribution from reciprocal space
  for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
  for (uint iExpB = 0; iExpB < pB->nFn; ++ iExpB)
  for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
  {
    double maxContraction = 0;
    for (uint iCoB = 0; iCoB < pB->nCo; ++ iCoB)
    for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
      double CoABC = pC->contractions(iExpC, iCoC) *
	pA->contractions(iExpA, iCoA) *
	pB->contractions(iExpB, iCoB);
      if (fabs(maxContraction) < fabs(CoABC))
	maxContraction = fabs(CoABC);
    }
    
    double
      a = pA->exponents[iExpA],
      b = pB->exponents[iExpB],
      alpha = a * b /(a + b),
      prefactor = Prefactor * PI11over2 / pow(a*b*c, 1.5) / pow(latsum.RVolume, 2);
    

    double scale = prefactor * pInv2A[iExpA] * pInv2B[iExpB]  * pInv2C[iExpC];
    

    for (int i=0; i<nterms; i++)
      pReciprocalSum[i] = 0.0;

    vector<double>& KLattice = latsum.KLattice;
    vector<double>& RLattice = latsum.RLattice;

    //this is the largest et2rho we will every encounter
    double eta2rho = latsum.Eta2RhoCoul; 
    int nloop = 0, gloop=0, validloop=0;
    
    for (int g=1; g<latsum.Kdist.size(); g++) {
      double expVal = pKernel->getValueKSpace(latsum.Kdist[g], 1.0, eta2rho);
      double Gx=latsum.Kcoord[3*g+0],  Gy=latsum.Kcoord[3*g+1],  Gz=latsum.Kcoord[3*g+2];
      double Gsq = latsum.Kdist[g];

      double sumOverF = 0.0;


      int fterms = 0;
      double logscreen = log(screen/expVal/scale/maxContraction);
      bool foundNonZero = false;

      for (int f=0; f<latsum.Kdist.size(); f++) {
	double Fx=latsum.Kcoord[3*f+0],  Fy=latsum.Kcoord[3*f+1],  Fz=latsum.Kcoord[3*f+2];
	double Fsq = Fx*Fx + Fy*Fy + Fz*Fz;
	double expArg = - (Gx*Fx+Gy*Fy+Gz*Fz)/2./a - b*Gsq/4./a/(a+b) - Fsq/4./alpha;

	if (expArg < logscreen) continue;
	foundNonZero = true;
	
	double expval2 = exp(expArg); 
	double maxG = getSphReciprocal3(la, lb, lc, pReciprocalSum, pSpha, pSphb, pSphc,
					Gx, Gy, Gz,
					Fx, Fy, Fz,
					TACx, TACy, TACz,
					TABx, TABy, TABz,
					expVal*expval2, scale);

	double checkScreen = expVal * expval2 * maxG * scale * maxContraction;
	sumOverF += checkScreen;
	nloop++; 
	if (checkScreen > screen)
	  validloop++;
	
	
	//if ( checkScreen < screen ) {
	  //exit(0);
	  //break;
	//}
      }
      gloop++;
      if (!foundNonZero)
	break;
    }

    cout <<"valid: "<<validloop<<" total: "<< nloop<<"  "<<gloop<<"  "<<rho<<"  "<<alpha<<"  "<<latsum.Kdist[1]<<endl;

    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
    for (uint iCoB = 0; iCoB < pB->nCo; ++ iCoB)
    for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
      double CoAC = pC->contractions(iExpC, iCoC) *
          pB->contractions(iExpB, iCoB) *
          pA->contractions(iExpA, iCoA);

      ct::Add2(&pOut[nterms*(iCoA + pA->nCo*(iCoB + pB->nCo*iCoC))],
           pReciprocalSum, CoAC, nterms);
    }

  }
  Mem.Free(pInt2A);

}
*/
void EvalInt2e3cKKsum(double *pOut, BasisShell const *pA,
		      BasisShell const *pB, BasisShell const *pC,
		      double Prefactor, Kernel *pKernel,
		      LatticeSum& latsum, ct::FMemoryStack2 &Mem) {

  double
    *pInv2A, *pInv2B, *pInv2C;
  Mem.Alloc(pInv2A, pA->nFn);
  Mem.Alloc(pInv2B, pB->nFn);  
  Mem.Alloc(pInv2C, pC->nFn);  
  for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
    pInv2A[iExpA] = bool(pA->l)? std::pow(1.0/(2*pA->exponents[iExpA]), (int)pA->l) : 1.;
  for (uint iExpB = 0; iExpB < pB->nFn; ++ iExpB)
    pInv2B[iExpB] = bool(pB->l)? std::pow(1.0/(2*pB->exponents[iExpB]), (int)pB->l) : 1.;
  for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
    pInv2C[iExpC] = bool(pC->l)? std::pow(1.0/(2*pC->exponents[iExpC]), (int)pC->l) : 1.;
  
  int la = pA->l, lb = pB->l, lc = pC->l;

  double TACx =  pA->Xcoord - pC->Xcoord,
    TACy =  pA->Ycoord - pC->Ycoord,
    TACz =  pA->Zcoord - pC->Zcoord;

  double TABx =  pA->Xcoord - pB->Xcoord,
    TABy =  pA->Ycoord - pB->Ycoord,
    TABz =  pA->Zcoord - pB->Zcoord;

  int nterms = (2*la+1)*(2*lb+1)*(2*lc+1);
  double *pReciprocalSum;
  Mem.Alloc(pReciprocalSum, nterms);
  
  double *pSpha, *pSphb, *pSphc;
  Mem.Alloc(pSpha, ((la+1)*(la+1)));
  Mem.Alloc(pSphb, ((lb+1)*(lb+1)));
  Mem.Alloc(pSphc, ((lc+1)*(lc+1)));
  

  double screen = latsum.Kscreen;
  double PI11over2 = pow(M_PI, 5.5);
  
  ksumTime2.start();
  //now go back to calculating contribution from reciprocal space
  for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
  for (uint iExpB = 0; iExpB < pB->nFn; ++ iExpB)
  for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
  {
    double maxContraction = 0;
    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
    for (uint iCoB = 0; iCoB < pB->nCo; ++ iCoB)
    for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
      double CoABC = pC->contractions(iExpC, iCoC) *
          pA->contractions(iExpA, iCoA) *
          pB->contractions(iExpB, iCoB);
      if (fabs(maxContraction) < fabs(CoABC))
        maxContraction = fabs(CoABC);
    }
    
    double
      a = pA->exponents[iExpA],
      b = pB->exponents[iExpB],
      c = pC->exponents[iExpC],
      alpha = a * b /(a + b),
      rho = (a+b)*c/(a+b+c),
      prefactor = Prefactor * PI11over2 / pow(a*b*c, 1.5) / pow(latsum.RVolume, 2);
      

    double scale = prefactor * pInv2A[iExpA] * pInv2B[iExpB]  * pInv2C[iExpC];
    

    for (int i=0; i<nterms; i++)
      pReciprocalSum[i] = 0.0;

    vector<double>& KLattice = latsum.KLattice;
    vector<double>& RLattice = latsum.RLattice;

    
    int nloop = 0;
    int gloop=0, validloop=0;
    //double logscreen = log(screen/scale/maxContraction);
    
    for (int g=1; g<latsum.Kdist.size(); g++) {
      //double expVal = kernel->getValueKSpace(latsum.Kdist[g], 1.0, Eta2rho);
      double expVal = pKernel->getValueKSpace(latsum.Kdist[g], 1.0, rho);
      double Gx=latsum.Kcoord[3*g+0],  Gy=latsum.Kcoord[3*g+1],  Gz=latsum.Kcoord[3*g+2];
      double Gsq = latsum.Kdist[g];

      double sumOverF = 0.0;


      int fterms = 0;
      double logscreen = log(screen/expVal/scale/maxContraction);
      bool foundNonZero = false;

      //find the nearest grid point
      double Fx0 = -b/(a+b)*Gx, Fy0 = -b/(a+b)*Gy, Fz0 = -b/(a+b)*Gz;
      int fx0 = std::round((RLattice[0]*Fx0 + RLattice[1]*Fy0 + RLattice[2]*Fz0)/2/M_PI);
      int fy0 = std::round((RLattice[3]*Fx0 + RLattice[4]*Fy0 + RLattice[5]*Fz0)/2/M_PI);
      int fz0 = std::round((RLattice[6]*Fx0 + RLattice[7]*Fy0 + RLattice[8]*Fz0)/2/M_PI);
      Fx0 = fx0*KLattice[0]+fy0*KLattice[3]+fz0*KLattice[6];
      Fy0 = fx0*KLattice[1]+fy0*KLattice[4]+fz0*KLattice[7];
      Fz0 = fx0*KLattice[2]+fy0*KLattice[5]+fz0*KLattice[8];

      for (int f=0; f<latsum.Kdist.size(); f++) {
	double Fx=Fx0+latsum.Kcoord[3*f+0],
	  Fy=Fy0+latsum.Kcoord[3*f+1],
	  Fz=Fz0+latsum.Kcoord[3*f+2];
	
	double Fsq = Fx*Fx + Fy*Fy + Fz*Fz;
	double expArg = - (Gx*Fx+Gy*Fy+Gz*Fz)/2./a - b*Gsq/4./a/(a+b) - Fsq/4./alpha;

	if (expArg < logscreen) continue;
	foundNonZero = true;
	
	double expval2 = exp(expArg); //this one does
	//if (Fx==0&&Fy==0&&Fz==0) expval2 /= 2.;

	double maxG = getSphReciprocal3(la, lb, lc, pReciprocalSum, pSpha, pSphb, pSphc,
					Gx, Gy, Gz,
					Fx, Fy, Fz,
					TACx, TACy, TACz,
					TABx, TABy, TABz,
					expVal*expval2, scale);

	double checkScreen = expVal * expval2 * maxG * scale * maxContraction;
	sumOverF += checkScreen;
	nloop++; 
	if (checkScreen > screen)
	  validloop++;
		
      }
      gloop++;
      if (!foundNonZero)
	break;
    }


    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
    for (uint iCoB = 0; iCoB < pB->nCo; ++ iCoB)
    for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
      double CoAC = pC->contractions(iExpC, iCoC) *
          pB->contractions(iExpB, iCoB) *
          pA->contractions(iExpA, iCoA);

      ct::Add2(&pOut[nterms*(iCoA + pA->nCo*(iCoB + pB->nCo*iCoC))],
           pReciprocalSum, CoAC, nterms);
    }
    //cout <<"valid: "<<validloop<<" total: "<< nloop<<"  gloop: "<<gloop<<"  "<<rho<<"  "<<alpha<<"  "<<latsum.Kdist[1]<<"  "<<pReciprocalSum[0]<<"  "<<pReciprocalSum[4]<<"  "<<pReciprocalSum[8]<<"  "<<pOut[0]<<"  "<<pOut[4]<<"  "<<pA->nCo<<"  "<<pB->nCo<<endl;

  }
  Mem.Free(pInv2A);
  
}


void ThreeCenterIntegrals(std::vector<int>& shls, BasisSet& basis, std::vector<double>& Lattice, ct::FMemoryStack2& Mem) {

  //10 along R and G directions
  LatticeSum latsum(&Lattice[0], 10, 10, Mem, basis, 1., 2., 1.e-11, 1e-11);
  cout <<"et2rho "<< latsum.Eta2RhoCoul<<endl;
  
  //first calculate the auxbasis G array and store it
  size_t nAuxbas = basis.getNPrimitivebas(shls[5]) - basis.getNPrimitivebas(shls[4]);

  //AuxGarray(c,G) -> c + G * nC
  double* AuxGarray;
  //Mem.ClearAlloc(AuxGarray, nAuxbas * latsum.Kcoord.size());

  auto start = high_resolution_clock::now();
  //PopulateAuxGMatrix(AuxGarray, basis, shls, latsum, Mem);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout <<"Auxbas populate "<< duration.count()/1e6<<"  size: "<<nAuxbas*latsum.Kcoord.size()<<"  "<<nAuxbas<<"  "<<latsum.Kcoord.size()<<"  "<<shls[0]<<"  "<<shls[1]<<"  "<<shls[2]<<"  "<<shls[3]<<"  "<<shls[4]<<"  "<<shls[5]<<endl;

  
  cout << basis.getNPrimitivebas(shls[5])<<"  "<<basis.getNbas(shls[5])<<endl;
  cout << basis.getNPrimitivebas(shls[4])<<"  "<<basis.getNbas(shls[4])<<endl;


  //nbasis
  size_t nbas = basis.getNbas(shls[1]) - basis.getNbas(shls[2]);
  vector<double> Integral3c(nbas*nbas*nAuxbas,0.0);
  ContractWithBasisPair(&Integral3c[0], AuxGarray, shls, basis, latsum, Mem);
}

void PopulateAuxGMatrix(double* pOut, BasisShell* pC, size_t offset,
			size_t nAuxBas, LatticeSum& latsum, ct::FMemoryStack2 &Mem) {

  int lc = pC->l; int nterms = 2*lc +1;
  
  double *pInv2C;
  Mem.Alloc(pInv2C, pC->nFn);  
  for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
    pInv2C[iExpC]=bool(lc) ? std::pow(1.0/(2*pC->exponents[iExpC]), (int)lc) : 1.;
  
  double maxCon = 0.;
  for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC)
    for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC) 
      if (maxCon < fabs(pC->contractions(iExpC, iCoC)))
	maxCon = fabs(pC->contractions(iExpC, iCoC));
  
  double* pSphc; Mem.Alloc(pSphc, 2*lc+1);
  double Cx =  pC->Xcoord,
    Cy =  pC->Ycoord,
    Cz =  pC->Zcoord;
  
  double screen = latsum.Kscreen;
  double Eta2Rho =  latsum.Eta2RhoCoul;
  
  double logscreen = log(1.e-20)-log(maxCon);
  
  for (int g=1; g<latsum.Kdist.size(); g++) {
    double Gx=latsum.Kcoord[3*g+0],
      Gy=latsum.Kcoord[3*g+1],
      Gz=latsum.Kcoord[3*g+2];
    
    
    if (-(Gx*Gx+Gy*Gy+Gz*Gz)/4/Eta2Rho < logscreen) break;
    ir::EvalSlcX_Deriv0(pSphc, Gx, Gy, Gz, lc);
      
    for (uint iExpC = 0; iExpC < pC->nFn; ++ iExpC) {
      if (-(Gx*Gx+Gy*Gy+Gz*Gz)/4/pC->exponents[iExpC] < logscreen) continue;
	
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
  
  double* pSpha1; Mem.Alloc(pSpha1, 2*la+1);
  double* pSpha2; Mem.Alloc(pSpha2, 2*la+1);
  double* pSphb1; Mem.Alloc(pSphb1, 2*lb+1);
  double* pSphb2; Mem.Alloc(pSphb2, 2*lb+1);
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
    
  for (int g=1; g<latsum.Kdist.size(); g++) {
    double Gx=latsum.Kcoord[3*g+0],
      Gy=latsum.Kcoord[3*g+1],
      Gz=latsum.Kcoord[3*g+2];
    double Gsq = latsum.Kdist[g];
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
  
  double* pSpha1; Mem.Alloc(pSpha1, 2*la+1);
  double* pSpha2; Mem.Alloc(pSpha2, 2*la+1);
  double* pSphb1; Mem.Alloc(pSphb1, 2*lb+1);
  double* pSphb2; Mem.Alloc(pSphb2, 2*lb+1);
  double Ax =  pA->Xcoord, Ay =  pA->Ycoord, Az =  pA->Zcoord;
  double Bx =  pB->Xcoord, By =  pB->Ycoord, Bz =  pB->Zcoord;

  
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
    
  double logscreen = log(screen) -
    log(maxContraction * scale);
    
  for (int g=1; g<latsum.Kdist.size(); g++) {
    double Gx=latsum.Kcoord[3*g+0],
      Gy=latsum.Kcoord[3*g+1],
      Gz=latsum.Kcoord[3*g+2];
    double Gsq = latsum.Kdist[g];
    double expArgG = -(Gsq)/4/min(a+b, Eta2Rho);
    
    if (expArgG < logscreen) break;
    bool foundNonZero = false;


    for (int i=0; i<nterms; i++) {
      pReciprocalSumCos[i] = 0.0;
      pReciprocalSumSin[i] = 0.0;
    }

    for (int q=0; q<latsum.Rdist.size(); q++) {
      double Qx = latsum.Rcoord[3*q+0],
	Qy = latsum.Rcoord[3*q+1],
	Qz = latsum.Rcoord[3*q+2];
      
      double Qsq = Qx*Qx + Qy*Qy + Qz*Qz;
      double expArgQ = -alpha * (    (Ax-Bx-Qx)*(Ax-Bx-Qx)
				  +  (Ay-By-Qy)*(Ay-By-Qy)
				     +  (Az-Bz-Qz)*(Az-Bz-Qz));
      
      if (expArgQ+expArgG < logscreen) break;

      double expval2 = exp(expArgQ);

      //pReciprocalSumCos[0] += expval2 * scale * cos(arg);
      //pReciprocalSumSin[0] += expval2 * scale * sin(arg);
      getSphRealRecursion(la, lb, pReciprocalSumCos,
			  pReciprocalSumSin,
			  Gx, Gy, Gz,
			  Qx, Qy, Qz, Ax, Ay, Az,
			  Bx, By, Bz, alpha, a, b, expval2,
			  scale, Mem);
      
    }
    if (!foundNonZero) break;
    ct::Add2(&pOutCos[g*nterms], pReciprocalSumCos, 1.0, nterms);
    ct::Add2(&pOutSin[g*nterms], pReciprocalSumSin, 1.0, nterms);
  }
  Mem.Free(pReciprocalSumCos);
  pairRTime.stop();
  
}

void contractCoulombKernel(double* pOut, double* AuxGarray,
			   double* OrbPairGMatrixcos, double* OrbPairGMatrixsin,
			   LatticeSum& latsum, BasisShell* pA, BasisShell *pB,
			   int iExpA, int iExpB,
			   int nAuxbas, BasisSet& basis, vector<int>& shls,
			   ct::FMemoryStack2 &Mem) {

  coulombContractTime.start();
  
  double Eta2Rho =  latsum.Eta2RhoCoul;
  double logscreen = log(latsum.Kscreen);
  int Pidx = 0;

  double a = pA->exponents[iExpA], b = pB->exponents[iExpB];
  int la = pA->l, lb = pB->l;
  int ntermsab = (2*la+1)*(2*lb+1);

  for (int shlc = shls[4]; shlc <shls[5]; shlc++) {
    BasisShell *pC = & basis.BasisShells[shlc];
    int lc = pC->l; int ntermsc = 2*lc+1; int nterms = ntermsab * ntermsc;
    
    double *pSphc; Mem.Alloc(pSphc, (lc+1)*(lc+1));
    double *pReciprocal; Mem.Alloc(pReciprocal, ntermsab*ntermsc);
    
    for (int iExpC = 0; iExpC < pC->nFn; iExpC++) {
      double c = pC->exponents[iExpC];
      double rho = min((a+b)*c/(a+b+c), Eta2Rho);
      double Cx = pC->Xcoord, Cy = pC->Ycoord, Cz = pC->Zcoord;
      double signCos = (lc%4 == 0 || lc%4==3 ) ?  1. : -1.; 
      double signSin = (lc%4 == 0 || lc%4==1 ) ?  1. : -1.;
      double prefactor = 1./pow(c, 1.5) * std::pow(1.0/(2*c), lc);

      for (int i=0; i<ntermsab*ntermsc; i++) pReciprocal[i] = 0.0;

      double sum = 0.0;
      for (int g=1; g<latsum.Kdist.size(); g++) {
	double Gx=latsum.Kcoord[3*g+0],
	  Gy=latsum.Kcoord[3*g+1],
	  Gz=latsum.Kcoord[3*g+2];
	double Gsq = latsum.Kdist[g];
	double expArgG = -(Gsq)/4/rho;

	if (expArgG < logscreen) break;
	double Gkernel = exp(expArgG)/(Gsq/4.);
	double Arg = Cx*Gx+Cy*Gy+Cz*Gz;
	ir::EvalSlcX_Deriv0(pSphc, Gx, Gy, Gz, lc);

	//(ab, P) = (P, G) (G)  (ab, G)
	for (int ab = 0; ab<ntermsab; ab++)
	  for (int c = 0; c<ntermsc; c++)
	    pReciprocal[ab+ntermsab*c] += pSphc[c] * Gkernel * prefactor *
	      ( signCos * cos(Arg) * OrbPairGMatrixcos[ab + g*ntermsab] +
		signSin * sin(Arg) * OrbPairGMatrixsin[ab + g*ntermsab]);
	sum += Gkernel*prefactor * OrbPairGMatrixcos[g];

      }

      //contract the primitive
      for (uint iCoC = 0; iCoC < pC->nCo; ++ iCoC)
      for (uint iCoB = 0; iCoB < pB->nCo; ++ iCoB)
      for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
	double CoAC = pC->contractions(iExpC, iCoC) *
          pB->contractions(iExpB, iCoB) *
          pA->contractions(iExpA, iCoA);
	
	ct::Add2(&pOut[nterms*(iCoA + pA->nCo*(iCoB + pB->nCo*iCoC))],
		 pReciprocal, CoAC, nterms);
      }
    }
    Mem.Free(pSphc);
  }
  coulombContractTime.stop();
}

void ContractWithBasisPair(double *pOut, double* AuxGarray, 
			   vector<int>& shls, BasisSet& basis, LatticeSum& latsum,
			   ct::FMemoryStack2 &Mem) {

  int nG = latsum.Kcoord.size();

  size_t nAuxbas = basis.getNbas(shls[5]) - basis.getNbas(shls[4]);

  double* primitiveInts; Mem.ClearAlloc(primitiveInts, nAuxbas * 13*13);
  
  cout << basis.BasisShells.size()<<"  nshells "<<endl;
  auto start = high_resolution_clock::now();
  for (int shla = shls[0]; shla < shls[1]; shla++) 
  for (int shlb = shls[2]; shlb <= shla; shlb++) {
    BasisShell *pA = &basis.BasisShells[shla];
    BasisShell *pB = &basis.BasisShells[shlb];    
    //cout <<"contractions "<< pA->contractions(0,0)<<"  "<<pB->contractions(0,0)<<"  "<<endl;
    cout <<"contractions "<< shla<<"  "<<shlb<<"  "<<pA->l<<"  "<<pB->l<<"  "<<endl;
    
    int la = pA->l, lb = pB->l; int ntermsab = (2*la+1)*(2*lb+1);
    double* OrbPairGMatrixCos; Mem.Alloc(OrbPairGMatrixCos, nG*ntermsab);
    double* OrbPairGMatrixSin; Mem.Alloc(OrbPairGMatrixSin, nG*ntermsab);

    for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA) 
    for (uint iExpB = 0; iExpB < pB->nFn; ++ iExpB) {
      double a = pA->exponents[iExpA], b = pB->exponents[iExpB];

      //cout << "about to populate pair "<<a<<"  "<<b<<endl;


      for (int g=0; g<nG*ntermsab; g++){
	OrbPairGMatrixCos[g] = 0.0;
	OrbPairGMatrixSin[g] = 0.0;
      }

      double alpha = a*b/(a+b);
      if (alpha < 0.1)
	PopulatePairGMatrixKspace(OrbPairGMatrixCos, OrbPairGMatrixSin,
				  pA, pB, iExpA, iExpB, latsum, Mem);
      else
	PopulatePairGMatrixRspace(OrbPairGMatrixCos, OrbPairGMatrixSin,
				  pA, pB, iExpA, iExpB, latsum, Mem);
      contractCoulombKernel(primitiveInts, AuxGarray, OrbPairGMatrixCos,
			    OrbPairGMatrixSin,
			    latsum, pA, pB, iExpA, iExpB, nAuxbas, basis, shls, Mem);

    }
    Mem.Free(OrbPairGMatrixCos);
    
  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout <<"Auxbas populate "<< duration.count()/1e6<<endl;
  cout << pairRTime<<"  "<<pairKTime<<"  "<<coulombContractTime<<endl;
  exit(0);
}



void PopulatePairGMatrix(double* pOut, BasisShell* pA, BasisShell* pB, 
			 size_t nFnPair, LatticeSum& latsum, ct::FMemoryStack2 &Mem) {

  int la = pA->l, lb = pB->l; int nterms = (2*la +1)*(2*lb+1);
  double* pReciprocalSum; Mem.Alloc(pReciprocalSum, nterms);

  double
    *pInv2A, *pInv2B;
  Mem.Alloc(pInv2A, pA->nFn);
  Mem.Alloc(pInv2B, pB->nFn);  
  for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA) 
    pInv2A[iExpA] = bool(la)? std::pow(1.0/(2*pA->exponents[iExpA]), (int)la) : 1.;
  
  for (uint iExpB = 0; iExpB < pB->nFn; ++ iExpB)
    pInv2B[iExpB] = bool(lb)? std::pow(1.0/(2*pB->exponents[iExpB]), (int)lb) : 1.;
  
  double* pSpha; Mem.Alloc(pSpha, 2*la+1);
  double* pSphb; Mem.Alloc(pSphb, 2*lb+1);
  double Ax =  pA->Xcoord, Ay =  pA->Ycoord, Az =  pA->Zcoord;
  double Bx =  pB->Xcoord, By =  pB->Ycoord, Bz =  pB->Zcoord;

  
  double screen = latsum.Kscreen;
  double Eta2Rho =  latsum.Eta2RhoCoul;
  double PI11over2 = pow(M_PI, 5.5);
  
  vector<double>& KLattice = latsum.KLattice;
  vector<double>& RLattice = latsum.RLattice;
  
  for (uint iExpB = 0; iExpB < pB->nFn; ++ iExpB)
  for (uint iExpA = 0; iExpA < pA->nFn; ++ iExpA)
  {
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
    double scale = prefactor * pInv2A[iExpA] * pInv2B[iExpB] ;
    
    double logscreen = log(screen) -
      log(maxContraction * scale);
    
    for (int g=1; g<latsum.Kdist.size(); g++) {
      double Gx=latsum.Kcoord[3*g+0],
	Gy=latsum.Kcoord[3*g+1],
	Gz=latsum.Kcoord[3*g+2];
      double Gsq = latsum.Kdist[g];
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

      for (int i=0; i<nterms; i++)
	pReciprocalSum[i] = 0.0;
      
      for (int f=0; f<latsum.Kdist.size(); f++) {
	double Fx=Fx0+latsum.Kcoord[3*f+0],
	  Fy=Fy0+latsum.Kcoord[3*f+1],
	  Fz=Fz0+latsum.Kcoord[3*f+2];
	
	double Fsq = Fx*Fx + Fy*Fy + Fz*Fz;
	double expArgF = - (Gx*Fx+Gy*Fy+Gz*Fz)/2./a - b*Gsq/4./a/(a+b) - Fsq/4./alpha;
	if (expArgF+expArgG < logscreen) continue;
	foundNonZero = true;

	double expval2 = exp(expArgF);
	if (Fx==0&&Fy==0&&Fz==0) expval2 /= 2.;

	getSphReciprocal2(la, lb, pReciprocalSum,
			  pSpha, pSphb, Gx, Gy, Gz,
			  Fx, Fy, Fz, Ax, Ay, Az,
			  Bx, By, Bz, expval2, 4.*scale);
		  
      }

      for (uint iCoB = 0; iCoB < pB->nCo; ++ iCoB)
	for (uint iCoA = 0; iCoA < pA->nCo; ++ iCoA) {
	  double CoAC = pB->contractions(iExpB, iCoB) *
	    pA->contractions(iExpA, iCoA);
	  
	  ct::Add2(&pOut[nterms*(iCoA + pA->nCo*iCoB) + g * nFnPair],
		   pReciprocalSum, CoAC, nterms);
	}

    }
  }
  Mem.Free(pReciprocalSum);
  
}

