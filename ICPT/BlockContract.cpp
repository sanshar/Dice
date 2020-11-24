/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/

#include <stdexcept>
#include <string.h> // for memset

#include "BlockContract.h"
#include "CxAlgebra.h"
#include "TensorTranspose.h"


#include <iostream>
#include <boost/format.hpp>
#ifdef _DEBUG
#include <iostream>
#include <boost/format.hpp>
// some printing stuff which may be helpful in debug mode.
// Not included in default mode because it induces unnecessary dependencies, and
// the host programs probably have own means of printing this kind of stuff anyway.

template<class T>
void IrPrintRow(std::ostream &xout, T const *pData, int iColSt, unsigned nCols, std::string const &FmtF, char const *pEndLn="\n")
{
   for (unsigned iCol = 0; iCol < nCols; ++ iCol)
      if ( FmtF != "" )
         xout << boost::format("%14.6f") % pData[iColSt * iCol];
      else
         xout << boost::format(FmtF) % pData[iColSt * iCol];
   xout << pEndLn;
};

void IrPrintMatrixGen(std::ostream &xout, double *pData, unsigned nRows, unsigned iRowSt, unsigned nCols, unsigned iColSt, std::string const &Caption)
{
   using boost::format;
   xout << format("  Matrix %s, %i x %i.\n") % Caption % nRows % nCols;
   std::string
      FmtS = "%14s",
      FmtI = "%11i   ",
      FmtF = "%14.6f";
   xout << format(FmtS) % "";
   for (unsigned i = 0; i < nCols; ++ i)
      xout << " " << format(FmtI) % i;
   xout << std::endl;
   for (unsigned iRow = 0; iRow < nRows; ++ iRow) {
      xout << format(FmtI) % iRow;
      IrPrintRow(xout, &pData[iRow*iRowSt], iColSt, nCols, FmtF);
   }
   xout << std::endl;
};
#endif // _DEBUG



// sanity check: assert that slots A[A0..An] have the same sizes as B[B0..Bn].
static void AssertSlotsCompatible(FNdArrayView const &A, uint A0, uint An,
                                  FNdArrayView const &B, uint B0, uint Bn)
{
   assert(An <= A.Rank());
   assert(Bn <= B.Rank());
   assert(An - A0 == Bn - B0);
   for (uint i = 0; i < Bn - B0; ++ i ) {
      assert(A.Sizes[A0+i] == B.Sizes[B0+i]);
   }
}

static void AssertCompatible(FNdArrayView const &A, FNdArrayView const &B)
{
   assert(A.Rank() == B.Rank());
   AssertSlotsCompatible(A, 0, A.Rank(), B, 0, B.Rank());
}

static FArrayOffset iStrideCostFn(FArrayOffset iStA, FArrayOffset iStB)
{
   if (iStA == 1 || iStB == 1)
      return 0;
   return iStA + iStB + std::max(iStA,iStB);
}
// ^- this doesn't make sure that we actually get the 1 stride as fastest
//    dimension, does it?

static void PermuteSlotPairs(FNdArrayView &A, uint A0, uint An,
                        FNdArrayView &B, uint B0, uint Bn)
{
   assert(An - A0 == Bn - B0);
   uint n = An - A0;
   for (uint i = 0; i < n; ++ i)
      for (uint j = i+1; j < n; ++ j)
         if (iStrideCostFn(A.Strides[A0+j], B.Strides[B0+j]) <
             iStrideCostFn(A.Strides[A0+i], B.Strides[B0+i]))
         {
            A.SwapSlots(A0+i, A0+j);
            B.SwapSlots(B0+i, B0+j);
         }
};


static void MakeLinearStrides0(FArraySizes &Strides, FArraySizes const &Sizes)
{
   Strides.push_back(1);
   for (uint i = 0; i != Sizes.size(); ++ i)
      Strides.push_back(Strides.back() * Sizes[i]);
};


// static void MakeLinearStrides1(FArraySizes &Out, FNdArrayView const &A, uint A0, uint An)
// {
//    for (uint i = A0; i != An; ++ i)
//       Out.push_back(Out.back() * A.Sizes[i]);
// };
//
//
// static void MakeLinearStrides2(FArraySizes &Out, FNdArrayView const &A,
//    bool Trans, uint nSlots0)
// {
//    Out.push_back(1);
//    if (!Trans) {
//       MakeLinearStrides1(Out, A, 0, nSlots0);
//       MakeLinearStrides1(Out, A, nSlots0, A.Rank());
//    } else {
//       MakeLinearStrides1(Out, A, nSlots0, A.Rank());
//       MakeLinearStrides1(Out, A, 0, nSlots0);
//    }
// }

static void PermuteSlotsAb(FNdArrayView &A, uint nSlots0)
{
   FArraySizes
      OldSizes = A.Sizes,
      OldStrides = A.Strides;
   A.Sizes.clear();
   A.Strides.clear();
   for (uint i = nSlots0; i != OldSizes.size(); ++ i){
      A.Sizes.push_back(OldSizes[i]);
      A.Strides.push_back(OldStrides[i]);
   }
   for (uint i = 0; i != nSlots0; ++ i){
      A.Sizes.push_back(OldSizes[i]);
      A.Strides.push_back(OldStrides[i]);
   }
};


static char TrCh(bool Trans) {
   return Trans? 'T' : 'N';
}

void ContractFirst(FNdArrayView D, FNdArrayView S, FNdArrayView T,
   FScalar Factor, bool Add, ct::FMemoryStack &Mem)
{
//    return ContractFirst_Naive(D,S,T,Factor,Add,Mem);
   uint
      nSlotsL = (S.Rank() + T.Rank() - D.Rank())/2,
      nSlotsA = S.Rank() - nSlotsL,
      nSlotsB = T.Rank() - nSlotsL;
   assert(2*nSlotsL + D.Rank() == S.Rank() + T.Rank());
   AssertSlotsCompatible(D,0,nSlotsA, S,nSlotsL,S.Rank());
   AssertSlotsCompatible(D,nSlotsA,D.Rank(), T,nSlotsL,T.Rank());
   AssertSlotsCompatible(S,0,nSlotsL, T,0,nSlotsL);

   // count the dimensions and transpositions.
   FArrayOffset
      nA = D.nValues(0, nSlotsA),
      nB = D.nValues(nSlotsA, nSlotsA+nSlotsB),
      nL = S.nValues(0, nSlotsL);

   if (nA == 0 || nB == 0 || nL == 0)
      // nothing to do here. And if we keep going on, MKL complains
      // about zero strides.
      return;

   // check out which matrix storage format conversion requires the
   // least jumping in memory. Effectively we need to decide on the
   // 'N'/'T' property of D/S/T, and on the simultaneous permutation
   // of A, B, and L indices.
   //
   // a good thing would be to do as little as possible to whatever
   // is the largest tensor. And in particular: if it is possible to
   // just let it stand, without actually copying it in memory (by
   // transposing the other tensors), this should be attempted.
   // This is always possible if the two index subsets are
   // non-interleaving.

   // But here we only do a simple greedy heuristic for a start [*]
   // - Whatever subset of D/S/T has the Strides[i] == 1 slot becomes
   //   the fast dimension. This decides on the Trans properties:

   bool
      TransD = true, // 'N': D[A,B],  'T': D[B,A]
      TransS = true, // 'N': S[L,A],  'T': S[A,L]
      TransT = true; // 'N': T[L,B],  'T': T[B,L]

   for (uint i = 0; i < nSlotsL; ++ i)
      if (S.Strides[i] == 1) TransS = false;
   for (uint i = 0; i < nSlotsL; ++ i)
      if (T.Strides[i] == 1) TransT = false;
   for (uint i = 0; i < nSlotsA; ++ i)
      if (D.Strides[i] == 1) TransD = false;

   // - Then the permutations on A/B/L are decided by ordering,
   //     fn(D.Strides[A], S.Strides[A]),
   //     fn(D.Strides[B], T.Strides[B]),
   //     fn(S.Strides[L], T.Strides[L]),
   //   where fn is some simple function of the two sub-arrays, still
   //   to be decided ('+', 'min' and 'max' all do the right thing in some
   //   simple cases. Needs to be tested experimentally I guess)
   //
   // [*] (in small cases full iteration through all the permutations
   //      and 8 TransX combinations would also be possible):
   // [**] I could also check for the largest directly MxM-able
   //      subset of the block contraction on the largest tensor (there
   //      always is one!) and never copy any data unless it is small
   //      (loop through the rest of the large tensor). Hard to say
   //      if that would help, though.

   PermuteSlotPairs(S,0,nSlotsL, T,0,nSlotsL);
   PermuteSlotPairs(S, nSlotsL, S.Rank(),  D, 0, nSlotsA);
   PermuteSlotPairs(T, nSlotsL, T.Rank(),  D, nSlotsA, D.Rank());

   // optionally switch A/B, L/A, L/B dimensions of D/S/T
   FArraySizes
      StridesMxmD, StridesMxmS, StridesMxmT;
   if (TransD) PermuteSlotsAb(D, nSlotsA);
   if (TransS) PermuteSlotsAb(S, nSlotsL);
   if (TransT) PermuteSlotsAb(T, nSlotsL);
   // make strides for matrix form of the contraction.
   MakeLinearStrides0(StridesMxmD, D.Sizes);
   MakeLinearStrides0(StridesMxmS, S.Sizes);
   MakeLinearStrides0(StridesMxmT, T.Sizes);

   // (copy required? Should check. Note, for example, that for
   // actual matrices (2d data) a copy is *never* required.)
   // note: I think that if the input data is continuous, a copy is
   //       not required iff the mxm strides are ascending or descending.
   //       They are certainly not requred if in ReorderData1A the
   //       input strides and the output strides are identical.
   void
      *pBaseOfMemory = Mem.Alloc(0);
   FScalar
      scalar,
      *pMatS = Mem.AllocN(nL * nA, scalar),
      *pMatT = Mem.AllocN(nL * nB, scalar),
      *pMatD = Mem.AllocN(nA * nB, scalar);
   FArrayOffset
      ldS = (!TransS)? nL : nA,
      ldT = (!TransT)? nL : nB,
      ldD = (!TransD)? nA : nB;

//    IrPrintMatrixGen(std::cout, S.pData, nL, 1, nA, nL, "ORIG DATA: S");
   ReorderData1A(pMatS, S.pData, StridesMxmS, S.Sizes, 1., false, true, &S.Strides);
//    IrPrintMatrixGen(std::cout, pMatS, nL, 1, nA, nL, "AFTER TRANSPOSE: S");
//    IrPrintMatrixGen(std::cout, T.pData, nL, 1, nB, nL, "ORIG DATA: T");
   ReorderData1A(pMatT, T.pData, StridesMxmT, T.Sizes, 1., false, true, &T.Strides);
//    IrPrintMatrixGen(std::cout, pMatT, nL, 1, nB, nL, "AFTER TRANSPOSE: T");


   if (!TransD) {
     FScalar scale1 = 1.0, scale0=0.0;
#ifdef _SINGLE_PRECISION
     
     sgemm_(TrCh(!TransS), TrCh(TransT), nA, nB, nL,
	  scale1, pMatS, ldS, pMatT, ldT, scale0, pMatD, ldD);
#else
     dgemm_(TrCh(!TransS), TrCh(TransT), nA, nB, nL,
	  scale1, pMatS, ldS, pMatT, ldT, scale0, pMatD, ldD);
#endif
   } else {
      // cannot directly feed that into dgemm, but note that
      // if D^T = A*B, then D = (A*B)^T = B^T * A^T, and this we can.
     FScalar scale1 = 1.0, scale0=0.0;
#ifdef _SINGLE_PRECISION
     sgemm_(TrCh(!TransT), TrCh(TransS), nB, nA, nL,
	  scale1, pMatT, ldT, pMatS, ldS, scale0, pMatD, ldD);
#else
     dgemm_(TrCh(!TransT), TrCh(TransS), nB, nA, nL,
	  scale1, pMatT, ldT, pMatS, ldS, scale0, pMatD, ldD);
#endif
   }

//    IrPrintMatrixGen(std::cout, pMatD, nA, 1, nB, nA, "AFTER MXM: D");
   ReorderData1A(D.pData, pMatD, D.Strides, D.Sizes, Factor, Add, true, &StridesMxmD);
//    ReorderData1A(pMatD, D.pData, StridesMxmD, D.Sizes, Factor, Add, false, &D.Strides);
//    IrPrintMatrixGen(std::cout, D.pData, nA, 1, nB, nA, "AFTER TRANSPOSE: D");
   Mem.Free(pBaseOfMemory);
}

// convert logical total array offset (linear strides, no holes) into
// physical total array offset (actual strides)
static size_t i2o(FNdArrayView const &A, FArrayOffset iLogical)
{
   FArrayOffset
      iOut = 0;
   for (uint iSlot = 0; iSlot < A.Rank(); ++ iSlot){
      iOut += A.Strides[iSlot] * (iLogical % A.Sizes[iSlot]);
      iLogical /= A.Sizes[iSlot];
   }
   return iOut;
};

void ContractFirst_Naive(FNdArrayView D, FNdArrayView S, FNdArrayView T,
   FScalar Factor, bool Add, ct::FMemoryStack &Mem)
{
   uint
      nSlotsL = (S.Rank() + T.Rank() - D.Rank())/2,
      nSlotsA = S.Rank() - nSlotsL,
      nSlotsB = T.Rank() - nSlotsL;
   FArrayOffset
      nA = D.nValues(0, nSlotsA),
      nB = D.nValues(nSlotsA, nSlotsA+nSlotsB),
      nL = S.nValues(0, nSlotsL);
   for (FArrayOffset iA = 0; iA < nA; ++ iA)
      for (FArrayOffset iB = 0; iB < nB; ++ iB)
      {
         FArrayOffset
            iOffD = i2o(D, iA + nA * iB);
         FScalar
            v = 0;
         if (Add)
            v = D.pData[iOffD];
         for (FArrayOffset iL = 0; iL < nL; ++ iL)
         {
            FArrayOffset
               iOffS = i2o(S, iL + nL * iA),
               iOffT = i2o(T, iL + nL * iB);
            v += Factor * S.pData[iOffS] * T.pData[iOffT];
         };
         D.pData[iOffD] = v;
      }
};


typedef TArrayFix<unsigned char, nMaxRank, unsigned char>
   FSlotIndices;

static void Append(FSlotIndices &Out, FSlotIndices const &A)
{
   for (uint i = 0; i < A.size(); ++ i)
      Out.push_back(A[i]);
};

static void FindCommonIndices(FSlotIndices &IA, FSlotIndices &IB,
   char const *pBegA, char const *pEndA, char const *pBegB, char const *pEndB)
{
   for (char const *pA = pBegA; pA != pEndA; ++ pA)
      for (char const *pB = pBegB; pB != pEndB; ++ pB)
         if (*pA == *pB) {
            IA.push_back(pA - pBegA);
            IB.push_back(pB - pBegB);
            break;
         }
};

static void PermuteSlots(FNdArrayView &D, FSlotIndices const &PermIn)
{
   FArraySizes
      OldSizes = D.Sizes,
      OldStrides = D.Strides;
   for (uint i = 0; i < PermIn.size(); ++ i) {
      D.Sizes[i] = OldSizes[PermIn[i]];
      D.Strides[i] = OldStrides[PermIn[i]];
   }
};

// static void PermuteSlotsDest(FNdArrayView &D, FSlotIndices const &PermIn)
// {
//    FArraySizes
//       OldSizes = D.Sizes,
//       OldStrides = D.Strides;
//    for (uint i = 0; i < PermIn.size(); ++ i) {
//       D.Sizes[PermIn[i]] = OldSizes[i];
//       D.Strides[PermIn[i]] = OldStrides[i];
//    }
// };

void ContractBinary(FNdArrayView D, char const *pDecl, FNdArrayView S, FNdArrayView T,
   FScalar Factor, bool Add, ct::FMemoryStack &Mem)
{
   // indices of A/B/L indices in D/S/T in input. Read from decl.
   FSlotIndices
      DA,DB, SL,SA, TL,TB;
   // find substrings corresponding to the D/S/T parts of the declaration.
   char const
      *pBegD, *pEndD, *pBegS, *pEndS, *pBegT, *pEndT,
      *p = pDecl;
   if (false) {
      // format:  S,T->D
      pBegS = p;
      for(; *p != 0; ++p)
         if (*p == ',') break;
      if (*p == 0) throw std::runtime_error("expected ',' in contraction declaration.");
      pEndS = p;
      ++ p;

      pBegT = p;
      for(; *p != 0; ++p)
         if (*p == '-') break;
      if (*p == 0 || *(p+1) != '>') throw std::runtime_error("expected '->' in contraction declaration.");
      pEndT = p;
      p += 2;

      pBegD = p;
      for(; *p != 0; ++p) {
      }
      pEndD = p;
   } else {
      // format:  D,S,T
      pBegD = p;
      for(; *p != 0; ++p)
         if (*p == ',') break;
      if (*p == 0) throw std::runtime_error("expected ',' in contraction declaration.");
      pEndD = p;
      ++ p;

      pBegS = p;
      for(; *p != 0; ++p)
         if (*p == ',') break;
      if (*p == 0) throw std::runtime_error("expected ',' in contraction declaration.");
      pEndS = p;
      ++ p;

      pBegT = p;
      for(; *p != 0; ++p) {
      }
      pEndT = p;
   }

   if (D.Rank() != pEndD - pBegD ||
       S.Rank() != pEndS - pBegS ||
       T.Rank() != pEndT - pBegT) {
      std::cout << boost::format("co-decl: '%s'  D: '%s'  S: '%s'  T: '%s'") % pDecl % std::string(pBegD, pEndD) % std::string(pBegS, pEndS) % std::string(pBegT, pEndT) << std::endl;
      throw std::runtime_error("ranks derived from contraction declaration differ from ranks of input tensors.");
   }

   // match up indices with all their counterparts
   FindCommonIndices(DA,SA, pBegD,pEndD, pBegS,pEndS);
   FindCommonIndices(DB,TB, pBegD,pEndD, pBegT,pEndT);
   FindCommonIndices(SL,TL, pBegS,pEndS, pBegT,pEndT);

   if (DA.size() + DB.size() != pEndD - pBegD ||
       SL.size() + SA.size() != pEndS - pBegS ||
       TL.size() + TB.size() != pEndT - pBegT)
      throw std::runtime_error(boost::str(boost::format("unsound index pattern in contraction declaration. Not a binary contraction: %s") % pDecl));
//       throw std::runtime_error("unsound index pattern in contraction declaration. Not a binary contraction");

   // re-order the tensors into D[A,B] += S[L,A] * T[L,B] form.
   FSlotIndices
      DAB, SLA, TLB;
   Append(DAB, DA); Append(DAB, DB); PermuteSlots(D, DAB);
   Append(SLA, SL); Append(SLA, SA); PermuteSlots(S, SLA);
   Append(TLB, TL); Append(TLB, TB); PermuteSlots(T, TLB);

   // relay to ContractFirst for the actual work.
   return ContractFirst(D, S, T, Factor, Add, Mem);
};

static int FindChr(char what, char const *pDecl){
   char const *p = pDecl;
   for ( ; *p != 0 && *p != ','; ++ p)
      if (*p == what) return p - pDecl;
   return -1;
}


std::string g_dbgPrefix = ""; // FIXME: REMOVE THIS

// note: (1) Declarations here come as 'dest,s0,s1,s2'. There is no '->', and dest comes first.
void ContractN(FNdArrayView **Ts, char const *pDecl, FScalar Factor, bool Add, ct::FMemoryStack &Mem)
{
   void *pBaseOfMemory = Mem.Alloc(0);
   bool Print = false;

   // find number of terms and boundaries between terms.
   uint const nMaxTerms = 32;
   uint nTerms = 1;
   uint iTerm[nMaxTerms];
   iTerm[0] = 0;
   {
      char const *p;
      for (p = pDecl; *p != 0; ++ p){
         if (*p == ',') {
            iTerm[nTerms] = p - pDecl + 1;
            nTerms += 1;
         }
      }
      iTerm[nTerms] = p - pDecl + 1;
   }

   if (Print) {
      std::cout << g_dbgPrefix << boost::format("co-decl: '%s'") % pDecl;
      for (uint i = 0; i < nTerms; ++ i)
         std::cout << boost::format("  %s[%i]: '%s'") % (i==0?'D':'S') % (i) % std::string(&pDecl[iTerm[i]], &pDecl[iTerm[i+1]]);
      std::cout << boost::format("  F: %.5f  Add: %i ") % Factor % Add << std::endl;
   }



   for (uint i = 0; i < nTerms; ++ i)
      if ( iTerm[i+1] - iTerm[i] - 1 != Ts[i]->Rank() )
         throw std::runtime_error("contraction decl/actual tensor dimension mismatch.");

   if (nTerms < 3)
      throw std::runtime_error("contractions with less than three terms not supported.");
   if (nTerms == 3) {
      return ContractBinary(*Ts[0], pDecl, *Ts[1], *Ts[2], Factor, Add, Mem);
   }

   // count number of distinct and common indices for all tensor pairs.
   // Find tensor pair with the smallest number of distinct indices
   // and the largest number of common indices.
   uint
      iBest = 0xffff, jBest = 0xffff;
   size_t
      DiMin = 0,
      CoMax = 0;
   for (uint i0 = 0; i0 < nTerms; ++ i0)
   for (uint i1 = i0+1; i1 < nTerms; ++ i1) {
      size_t
         Di = 1,
         Co = 1;
      for (uint s0 = 0; s0 != Ts[i0]->Rank(); ++ s0)
         if (-1 != FindChr(pDecl[iTerm[i0]+s0], &pDecl[iTerm[i1]])) {
            // common
            Co *= Ts[i0]->Sizes[s0];
//             if (Ts[i0]->Sizes[s0] != Ts[i1]->Sizes[s1])
//                throw std::runtime_error(boost::format("tensor sizes not consistent: %s") % pDecl);
         } else {
            // distinct: On T[i0], but not on T[i1]
            Di *= Ts[i0]->Sizes[s0];
         }
      for (uint s1 = 0; s1 != Ts[i1]->Rank(); ++ s1)
         if (-1 == FindChr(pDecl[iTerm[i1]+s1], &pDecl[iTerm[i0]]))
            // distinct: On T[i1], but not on T[i0]
            Di *= Ts[i1]->Sizes[s1];
      if (iBest == 0xffff || Di < DiMin || (Di == DiMin && Co > CoMax)) {
//          if (iBest == 0) continue; // FIXME: REMOVE THIS.
         iBest = i0;
         jBest = i1;
         DiMin = Di;
         CoMax = Co;
      }
   }

   // make new declaration string: remove i0 and i1, and replace by its
   // distinct indices.
   char
      *pDecl1,
      *pDecl2,
      *pDistinct;
   FNdArrayView **Ts1, **Ts2;
   uint nTs1 = 0, nDistinct = 0;
   Mem.Alloc(pDecl1, iTerm[nTerms]+1);
   Mem.Alloc(pDecl2, iTerm[nTerms]+1);
   Mem.Alloc(pDistinct, iTerm[nTerms]+1);
   Mem.Align(32);
   Mem.Alloc(Ts1, nTerms);
   Mem.Alloc(Ts2, nTerms);
   FNdArrayView
      Tmp;
   char *q = pDecl2, *p = pDecl1;
   for (uint i = 0; i < nTerms; ++ i) {
      if (i == iBest) {
         // put in the distinct indices between iBest and jBest.
         uint i0 = iBest, i1 = jBest;
         for (uint s0 = 0; s0 != Ts[i0]->Rank(); ++ s0)
            if (-1 == FindChr(pDecl[iTerm[i0]+s0], &pDecl[iTerm[i1]])) {
               pDistinct[nDistinct] = pDecl[iTerm[i0]+s0];
               *p++ = pDistinct[nDistinct];
               ++ nDistinct;
               Tmp.Sizes.push_back(Ts[i0]->Sizes[s0]);
            }
         for (uint s1 = 0; s1 != Ts[i1]->Rank(); ++ s1)
            if (-1 == FindChr(pDecl[iTerm[i1]+s1], &pDecl[iTerm[i0]])) {
//                *p++ = pDecl[iTerm[i1]][s1];
               pDistinct[nDistinct] = pDecl[iTerm[i1]+s1];
               *p++ = pDistinct[nDistinct];
               ++ nDistinct;
               Tmp.Sizes.push_back(Ts[i1]->Sizes[s1]);
            }
         Ts1[nTs1] = &Tmp;
         nTs1 += 1;
      } else if (i == jBest) {
         continue;
      } else {
         // nothing special: replicate original indices.
         for (uint s0 = 0; s0 != Ts[i]->Rank(); ++ s0)
            *p++ = pDecl[iTerm[i]+s0];
         Ts1[nTs1] = Ts[i];
         nTs1 += 1;
      }
      p[0] = ',';
      ++p;
   }
   -- p;
   p[0] = 0; // remove the trailing ','.

   if (Print) {
      std::cout << g_dbgPrefix << boost::format("co-pair: i = %i  j = %i  distinct: '%s'   Co = %i  Di = %i")
         % iBest % jBest % std::string(pDistinct,pDistinct+nDistinct) % CoMax % DiMin << std::endl;
   }

   // allocate intermediate contraction result.
//    Tmp.Strides.push_back(1);
   Tmp.Strides.resize(Tmp.Sizes.size()+1);
   Tmp.Strides[0] = 1;
//    for (uint i = 1; i < Tmp.Rank(); ++ i)
//       Tmp.Strides[i] = Tmp.Strides[i-1] * Tmp.Sizes[i-1];
//    Mem.Alloc(Tmp.pData, Tmp.nValues());
   for (uint i = 0; i < Tmp.Rank(); ++ i)
      Tmp.Strides[i+1] = Tmp.Strides[i] * Tmp.Sizes[i];
   Mem.Alloc(Tmp.pData, Tmp.Strides[Tmp.Rank()]);
   assert(Tmp.Strides[Tmp.Rank()] == Tmp.nValues());

   if (iBest == 0) {
      // factorization involves output tensor. need to make other stuff
      // first (Tmp) and then contract current result to final output.
      assert(Ts1[0] == &Tmp);
//       std::string s0 = g_dbgPrefix;
//       g_dbgPrefix = s0 + "RHS ";
      ContractN(Ts1, pDecl1, 1., false, Mem);
      Ts2[0] = Ts[iBest];
      Ts2[1] = Ts[jBest];
      Ts2[2] = &Tmp;
      for (uint s0 = 0; s0 < Ts2[0]->Rank(); ++ s0) *q++ = pDecl[iTerm[iBest]+s0]; *q++ = ',';
      for (uint s0 = 0; s0 < Ts2[1]->Rank(); ++ s0) *q++ = pDecl[iTerm[jBest]+s0]; *q++ = ',';
      for (uint s0 = 0; s0 < Ts2[2]->Rank(); ++ s0) *q++ = pDistinct[s0];
      q[0] = 0; // null-terminate.
//       g_dbgPrefix = s0 + "TMP ";
      ContractN(Ts2, pDecl2, Factor, Add, Mem);
//       g_dbgPrefix = s0;
   } else {
      // factorization on rhs only.
      assert(Ts1[0] != &Tmp);
      Ts2[0] = &Tmp;
      Ts2[1] = Ts[iBest];
      Ts2[2] = Ts[jBest];
      for (uint s0 = 0; s0 < Ts2[0]->Rank(); ++ s0) *q++ = pDistinct[s0]; *q++ = ',';
      for (uint s0 = 0; s0 < Ts2[1]->Rank(); ++ s0) *q++ = pDecl[iTerm[iBest]+s0]; *q++ = ',';
      for (uint s0 = 0; s0 < Ts2[2]->Rank(); ++ s0) *q++ = pDecl[iTerm[jBest]+s0];
      q[0] = 0; // null-terminate.
      std::string s0 = g_dbgPrefix;
//       g_dbgPrefix = s0 + "TMP ";
      ContractN(Ts2, pDecl2, 1., false, Mem);
//       g_dbgPrefix = s0 + "RST ";
      ContractN(Ts1, pDecl1, Factor, Add, Mem);
//       g_dbgPrefix = s0;
   }

   Mem.Free(pBaseOfMemory);
}


void ContractN(FNdArrayView Out, char const *pDecl, FScalar Factor, bool Add, ct::FMemoryStack &Mem,
   FNdArrayView *pT1, FNdArrayView *pT2, FNdArrayView *pT3, FNdArrayView *pT4,  FNdArrayView *pT5,
   FNdArrayView *pT6, FNdArrayView *pT7, FNdArrayView *pT8)
{
   FNdArrayView *pTs[9] = {&Out, pT1, pT2, pT3, pT4, pT5, pT6, pT7, pT8};
   ContractN(pTs, pDecl, Factor, Add, Mem);
}

void ContractN(FNdArrayView Out, char const *pDecl, FScalar Factor, bool Add, ct::FMemoryStack &Mem,
   FNdArrayView const &T1, FNdArrayView const &T2, FNdArrayView const &T3, FNdArrayView const &T4,
   FNdArrayView const &T5, FNdArrayView const &T6, FNdArrayView const &T7, FNdArrayView const &T8)
{
#define XCONV(TN) const_cast<FNdArrayView*>(&TN)
   FNdArrayView *pTs[9] = {&Out, XCONV(T1), XCONV(T2), XCONV(T3), XCONV(T4), XCONV(T5), XCONV(T6), XCONV(T7), XCONV(T8)};
   ContractN(pTs, pDecl, Factor, Add, Mem);
#undef XCONV
}


void Copy(FNdArrayView &Out, FNdArrayView const &In)
{
   // NOTE: if this becomes a problem, do it in a slightly less insanely
   // stupid manner. i2o was not actually intended to be used in real code.
  //AssertCompatible(Out, In);
   FArrayOffset
      N = Out.nValues();
   for (size_t i = 0; i != N; ++ i)
      Out.pData[i2o(Out,i)] = In.pData[i2o(In,i)];
};

void FNdArrayView::ClearData()
{
   // FIXME: will do the wrong thing (tm) if used on a non-continuous tensor.
   FArrayOffset
      TrialStride = 1;
   for (uint i = 0; i < Rank(); ++ i){
      // check for each stride which would occur in linear addressing if
      // it occurs somewhere in the actual set of strides.
      // (nah.. that's not really right, is it? but will work for direct
      //  linear storage pattern, so I leave it for the moment)
      bool Found = false;
      for (uint j = 0; j < Rank(); ++ j)
         if (Strides[i] == TrialStride)
            Found = true;
      if (!Found)
         throw std::runtime_error("ClearData() not implemented for non-continuous tensors.");
      TrialStride *= Sizes[i];
   }
   memset(pData, 0, nValues() * sizeof(pData[0]));
};



