/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#ifndef ITF_BLOCK_CONTRACT_H
#define ITF_BLOCK_CONTRACT_H

#include <stddef.h> // for size_t
#include <algorithm>
#include "CxDefs.h"
#include "CxFixedSizeArray.h"
#include "CxMemoryStack.h"

typedef unsigned int
   uint;

uint const
//    nMaxRank = 8;
   nMaxRank = 14;

typedef size_t FArrayOffset;

typedef TArrayFix<FArrayOffset, nMaxRank+1>
   FArraySizes;

// note: those things are supposed to be continuous in memory, even if
// they are transposed (i.e., no holes)
struct FNdArrayView
{
   FScalar
      *pData;
   FArraySizes
      // strides for each dimension. Might contain an additional element holding
      // the total number of values.
      Strides,
      // sizes in each dimension.
      Sizes;
   uint Rank() const { assert(Strides.size() >= Sizes.size()); return Sizes.size(); }

   inline FArrayOffset nValues(uint iSlot0, uint iSlotsN) const;
   inline void SwapSlots(uint iSlot0, uint iSlot1);
   inline FArrayOffset nValues() const { return nValues(0, Sizes.size()); }

   void ClearData();

   // yes, I know.
   inline FScalar &operator () (FArrayOffset i)
      { assert(Rank() == 1); return pData[i * Strides[0]]; };
   inline FScalar &operator () (FArrayOffset i, FArrayOffset j)
      { assert(Rank() == 2); return pData[i * Strides[0] + j*Strides[1]]; };
   inline FScalar &operator () (FArrayOffset i, FArrayOffset j, FArrayOffset k)
      { assert(Rank() == 3); return pData[i * Strides[0] + j*Strides[1] + k*Strides[2]]; };
   inline FScalar &operator () (FArrayOffset i, FArrayOffset j, FArrayOffset k, FArrayOffset l)
      { assert(Rank() == 4); return pData[i * Strides[0] + j*Strides[1] + k*Strides[2] + l*Strides[3]]; };
   inline FScalar &operator () (FArrayOffset i, FArrayOffset j, FArrayOffset k, FArrayOffset l, FArrayOffset m)
      { assert(Rank() == 5); return pData[i * Strides[0] + j*Strides[1] + k*Strides[2] + l*Strides[3] + m*Strides[4]]; };
   inline FScalar &operator () (FArrayOffset i, FArrayOffset j, FArrayOffset k, FArrayOffset l, FArrayOffset m, FArrayOffset n)
      { assert(Rank() == 6); return pData[i * Strides[0] + j*Strides[1] + k*Strides[2] + l*Strides[3] + m*Strides[4] + n*Strides[5]]; };
   inline FScalar &operator () (FArrayOffset i, FArrayOffset j, FArrayOffset k, FArrayOffset l, FArrayOffset m, FArrayOffset n, FArrayOffset o)
      { assert(Rank() == 7); return pData[i * Strides[0] + j*Strides[1] + k*Strides[2] + l*Strides[3] + m*Strides[4] + n*Strides[5] + o*Strides[6]]; };
   inline FScalar &operator () (FArrayOffset i, FArrayOffset j, FArrayOffset k, FArrayOffset l, FArrayOffset m, FArrayOffset n, FArrayOffset o, FArrayOffset p)
      { assert(Rank() == 8); return pData[i * Strides[0] + j*Strides[1] + k*Strides[2] + l*Strides[3] + m*Strides[4] + n*Strides[5] + o*Strides[6] + p*Strides[7]]; };
};


// perform contraction:
//      Dest[AB] += f * \sum_L S[LA] T[LB],
// where L runs over the first nCo slots of S, and of T, and the output is
// ordered as first A, then B.
// This function respects the strides arguments and can thus perform abitrary
// contractions as kernel routine. nCo is implicit: it is (S.Rank() + T.Rank() - Dest.Rank())/2.
void ContractFirst(FNdArrayView D, FNdArrayView S, FNdArrayView T,
   FScalar Factor, bool Add, ct::FMemoryStack &Mem);

// do the same as ContractFirst, but explicitly loop over stuff.
void ContractFirst_Naive(FNdArrayView D, FNdArrayView S, FNdArrayView T,
   FScalar Factor, bool Add, ct::FMemoryStack &Mem);

// Decl: A string denoting the index associations:
//       'abij,acik,cbkj' means D[abij] = \sum_kc S[acik] T[cbkj].
// Notes:
//    - Self-contractions (traces) or contractions involving more than two
//      source tensors are not allowed. This is for binary contractions only
//    - Index patterns must be sound (each index must occur exactly two times)
//    - Spaces in the declaration are not allowed. Apart from ',', '-' and '>',
//      all other characters are treated simply as placeholder symbols without
//      any particular meaning attached. That is, a pattern like "%.@,:%.->:@"
//      is fine.
void ContractBinary(FNdArrayView D, char const *pDecl, FNdArrayView S, FNdArrayView T,
   FScalar Factor, bool Add, ct::FMemoryStack &Mem);

void ContractN(FNdArrayView **Ts, char const *pDecl, FScalar Factor, bool Add, ct::FMemoryStack &Mem);
// these here are convenience functions with some runtime overhead.
// They just call the upper ContractN.
void ContractN(FNdArrayView Out, char const *pDecl, FScalar Factor, bool Add, ct::FMemoryStack &Mem,
   FNdArrayView const &T1, FNdArrayView const &T2 = FNdArrayView(), FNdArrayView const &T3 = FNdArrayView(), FNdArrayView const &T4 = FNdArrayView(),
   FNdArrayView const &T5 = FNdArrayView(), FNdArrayView const &T6 = FNdArrayView(), FNdArrayView const &T7 = FNdArrayView(), FNdArrayView const &T8 = FNdArrayView());
void ContractN(FNdArrayView Out, char const *pDecl, FScalar Factor, bool Add, ct::FMemoryStack &Mem,
   FNdArrayView *pT1, FNdArrayView *pT2 = 0, FNdArrayView *pT3 = 0, FNdArrayView *pT4 = 0,  FNdArrayView *pT5 = 0,
   FNdArrayView *pT6 = 0, FNdArrayView *pT7 = 0, FNdArrayView *pT8 = 0);


void Copy(FNdArrayView &Out, FNdArrayView const &In);

FArrayOffset FNdArrayView::nValues(uint iSlot0, uint iSlotsN) const
{
   assert(iSlot0 <= iSlotsN && iSlotsN <= this->Rank());
   FArrayOffset r = 1;
   for (uint i = iSlot0; i < iSlotsN; ++ i)
      r *= Sizes[i];
   return r;
};

void FNdArrayView::SwapSlots(uint iSlot0, uint iSlot1)
{
   std::swap(this->Strides[iSlot0], this->Strides[iSlot1]);
   std::swap(this->Sizes[iSlot0], this->Sizes[iSlot1]);
}


#endif // ITF_BLOCK_CONTRACT_H
