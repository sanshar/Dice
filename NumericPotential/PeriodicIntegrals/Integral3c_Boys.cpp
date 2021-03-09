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
#include "LatticeSum.h"
#include "interface.h"
#include "Integral3c_Boys.h"
#include "CxAlgebra.h"

using namespace std;
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
