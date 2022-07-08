/* Copyright (c) 2012  Gerald Knizia
 * 
 * This file is part of the IR/WMME program
 * (See https://sites.psu.edu/knizia/)
 * 
 * IR/WMME is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 * 
 * IR/WMME is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with bfint (LICENSE). If not, see http://www.gnu.org/licenses/
 */

#include <stdexcept>
#include <sstream>
#include <new>
#include <algorithm> // for std::min


#include "CxMemoryStack.h"
//#include "CxDefs.h"
//#include "CxOpenMpProxy.h"

// extra function to allow overwriting from the outside, and to allow automatic GDB breakpoints.
void CxCrashAfterMemoryError(char const *pMsg)
{
//     DEBUG_BREAK;
    throw std::runtime_error(pMsg);
}

namespace ct {

// note: boundary must be size-of-2
inline char *AlignCharPtr(char *p, size_t Boundary) {
//    size_t
//       iPos = reinterpret_cast<size_t>(p),
//       iNew = ((iPos - 1) | (Boundary - 1)) + 1;
//    return reinterpret_cast<char*>(iNew);
   return reinterpret_cast<char*>(AlignSizeT(reinterpret_cast<size_t>(p), Boundary));
}

inline void FMemoryStack2::PushTopMark() {
#ifdef _DEBUG
  //assert(m_Pos < m_Size - 8);
   *reinterpret_cast<size_t*>(&m_pDataAligned[m_Pos]) = 0xbadc0de;
   m_Pos += sizeof(size_t);
#endif
}
inline void FMemoryStack2::PopTopMark() {
#ifdef _DEBUG
   m_Pos -= sizeof(size_t);
   //assert(m_Pos < m_Size - 8);
   if ( *reinterpret_cast<size_t*>(&m_pDataAligned[m_Pos]) != 0xbadc0de ) {
      printf("\n\n\n===================\nstack inconsistent!\n===================\n");
      CxCrashAfterMemoryError("stack error");
   }
#endif
}

void* FMemoryStack2::Alloc(size_t nSizeInBytes)
{
   PopTopMark();
   size_t OldPos = m_Pos;
   m_Pos += AlignSizeT(nSizeInBytes, CX_DEFAULT_MEM_ALIGN);
   PushTopMark();
   if ( m_Pos >= m_Size )
      CxCrashAfterMemoryError("FMemoryStack2: Stack size exceeded.");
   return &m_pDataAligned[OldPos];
}

void FMemoryStack2::Free(void *p)
{
   PopTopMark();
   ptrdiff_t
      n = &m_pDataAligned[m_Pos] - (char*)p;
   if ( n < 0 )
      CxCrashAfterMemoryError("FMemoryStack2: Release address too low!");
   m_Pos -= n;
   // note: new start of heap should be aligned automagically, since p is required to come out of a aligned Alloc() call.
   PushTopMark();
}

size_t FMemoryStack2::MemoryLeft() {
   return m_Size - m_Pos;
}


FMemoryStack2::FMemoryStack2(char *pBase, size_t nActualSize, size_t nNeededSize)
   : m_pDataStart(0), m_pDataAligned(0)
{
   if (pBase && nActualSize >= nNeededSize)
      AssignMemory(pBase, nActualSize);
   else
      Create((nNeededSize != 0)? nNeededSize : nActualSize);
}

void FMemoryStack2::Create(size_t nSize)
{
   //assert_rt(m_pDataStart == 0);
   m_Size = nSize;
   m_Pos = 0;
   m_pDataStart = 0;
   m_bOwnMemory = false;
   if ( nSize != 0 ) {
      m_pDataStart = new(std::nothrow) char[nSize];
      m_bOwnMemory = true;
      if ( m_pDataStart == 0 ) {
         std::stringstream str; str.precision(2); str.setf(std::ios::fixed);
         str << "FMemoryStack2: Sorry, failed to allocate " << (static_cast<double>(nSize)/static_cast<double>(1ul<<20)) << " MB of memory.";
         CxCrashAfterMemoryError(str.str().c_str());
      }

    if (nSize < 2*CX_DEFAULT_MEM_ALIGN)
        CxCrashAfterMemoryError("FMemoryStack2: assigned workspace is too small.");
    m_pDataAligned = AlignCharPtr(m_pDataStart, CX_DEFAULT_MEM_ALIGN);
    m_Size -= m_pDataAligned - m_pDataStart;
    PushTopMark();
   }
}

void FMemoryStack2::AssignMemory(char *pBase_, size_t nSize)
{
   //assert(m_pDataStart == 0 && pBase_ != 0);
   Create(0);
   m_bOwnMemory = false;
   m_pDataStart = pBase_;
   m_Size = nSize;
   m_Pos = 0;
   if (nSize < 2*CX_DEFAULT_MEM_ALIGN)
      CxCrashAfterMemoryError("FMemoryStack2: assigned workspace is too small.");
   m_pDataAligned = AlignCharPtr(m_pDataStart, CX_DEFAULT_MEM_ALIGN);
   m_Size -= m_pDataAligned - m_pDataStart;
   PushTopMark();
}

void FMemoryStack2::Destroy()
{
#ifdef _DEBUG
   if ( m_Size != 0 )
      PopTopMark();
   else
      //assert(m_pDataStart == 0);
   //assert(m_Pos == 0);
#endif // _DEBUG
   if ( m_bOwnMemory )
      delete []m_pDataStart;
   m_pDataStart = 0;
   m_pDataAligned = 0;
}


FMemoryStack::~FMemoryStack()
{
}

FMemoryStack2::~FMemoryStack2()
{
   if ( m_bOwnMemory )
      delete []m_pDataStart;
   m_pDataStart = (char*)0xbadc0de;
   m_pDataAligned = (char*)0xbadc0de;
}

void FMemoryStack2::Align(uint Boundary)
{
   //assert(m_pDataAligned == AlignCharPtr(m_pDataAligned, CX_DEFAULT_MEM_ALIGN));
   PopTopMark();
   if (Boundary <= CX_DEFAULT_MEM_ALIGN) {
       m_Pos = AlignSizeT(m_Pos, Boundary);
   } else {
       m_Pos = AlignCharPtr(&m_pDataAligned[m_Pos], Boundary) - m_pDataAligned;
   }
   PushTopMark();
}

void FMemoryStack::Align(uint Boundary)
{
   size_t
      iPos = reinterpret_cast<size_t>(Alloc(0)),
      iNew = ((iPos - 1) | (Boundary - 1)) + 1;
   Alloc(iNew - iPos);
}


#ifdef MOLPRO
void* FMemoryStackMolproCore::Alloc( size_t nSizeInBytes )
{
//     return itf::MolproAlloc(nSizeInBytes);
    void *p = itf::MolproAlloc(nSizeInBytes);
    m_pPeak = std::max(m_pPeak, static_cast<char*>(p) + nSizeInBytes);
    return p;
}

void FMemoryStackMolproCore::Free(void *p)
{
    return itf::MolproFree(p);
}

size_t FMemoryStackMolproCore::MemoryLeft()
{
    return itf::MolproGetFreeMemory();
}

FMemoryStackMolproCore::FMemoryStackMolproCore()
    : m_pPeak(0)
{
    m_pInitialTop = GetTop();
    m_pPeak = m_pInitialTop;
}

FMemoryStackMolproCore::~FMemoryStackMolproCore()
{
}
#endif // MOLPRO






FMemoryStackArray::FMemoryStackArray( FMemoryStack &BaseStack )
    : pBaseStack(&BaseStack)
{
    nThreads = omp_get_max_threads();
    pSubStacks = new FMemoryStack2[nThreads];
    size_t
        nSize = static_cast<size_t>(0.98 * static_cast<double>(pBaseStack->MemoryLeft() / nThreads));

    pBaseStack->Alloc(pStackBase, nThreads * nSize);
    for (size_t i = 0; i < nThreads; ++ i)
        pSubStacks[i].AssignMemory(pStackBase + i * nSize, nSize);
}

void FMemoryStackArray::Release()
{
    //assert(pBaseStack != 0);
    delete []pSubStacks;
    pBaseStack->Free(pStackBase);
    pStackBase = 0;
    pBaseStack = 0;
}

FMemoryStackArray::~FMemoryStackArray()
{
    if (pBaseStack)
        Release();
}

FMemoryStack2 &FMemoryStackArray::GetStackOfThread()
{
    int iThread = omp_get_thread_num();
    //assert(iThread < (int)nThreads);
    return pSubStacks[iThread];
}


double *FOmpAccBlock::pTls() {
   return &m_pTlsData[omp_get_thread_num() * m_nAlignedSize];
}

void FOmpAccBlock::Init(double *pTarget, size_t nSize, unsigned Flags, FMemoryStack &Mem)
{
   //assert(m_nSize == 0);
   if (pTarget == 0)
       nSize = 0;
   m_nSize = nSize;
   m_nAlignedSize = AlignSizeT(nSize, CX_DEFAULT_MEM_ALIGN/sizeof(*m_pTarget));
   m_Flags = Flags;
   if (m_nSize != 0)
      m_pTarget = pTarget;
   else {
      m_pTarget = 0;
      m_nAlignedSize = 0;
      m_pTlsData = 0;
   }
   if (!m_pTarget)
      return;
   m_nThreads = omp_get_max_threads();
   Mem.Alloc(m_pTlsData, nTotalSize());

   // clear thread-local data. Or should I align it with the thread ids? may be better for cache purposes.
   #pragma omp parallel for
   for (int iBlock = 0; iBlock < int(nScalarBlocks(nTotalSize())); ++ iBlock) {
      double
         *pBlock = &m_pTlsData[nScalarBlockSize() * size_t(iBlock)],
         *pBlockEnd = std::min(pBlock + nScalarBlockSize(), m_pTlsData + nTotalSize());
      memset(pBlock, 0, (pBlockEnd-pBlock)*sizeof(*pBlock));
   }
}

static void Add2( double *r, double const *x, double f, std::size_t n )
{
   std::size_t
      i = 0;
//    for ( ; i < (n & (~3)); i += 4 ) {
//       r[i]   += f * x[i];
//       r[i+1] += f * x[i+1];
//       r[i+2] += f * x[i+2];
//       r[i+3] += f * x[i+3];
//    }
   for ( ; i < n; ++ i ) {
      r[i] += f * x[i];
   }
}


void FOmpAccBlock::Join()
{
   if (!m_pTarget) return;

   // horizontal sum. in this loop, each block is summed up across the previous processor dimension.
   #pragma omp parallel for
   for (int iBlock = 0; iBlock < int(nScalarBlocks(m_nSize)); ++ iBlock) {
      size_t
         iBlockOffs = nScalarBlockSize() * size_t(iBlock);
      double
         *pBlock = &m_pTlsData[iBlockOffs],
         *pBlockEnd = std::min(pBlock + nScalarBlockSize(), m_pTlsData + m_nSize);
      if (m_Flags & OMPACC_ClearTarget)
         memset(&m_pTarget[iBlockOffs], 0, sizeof(*pBlock)*(pBlockEnd-pBlock));

      for (size_t iAcc = 0; iAcc < m_nThreads; ++ iAcc)
         Add2(&m_pTarget[iBlockOffs], &pBlock[iAcc * m_nAlignedSize], 1.0, pBlockEnd-pBlock);
   }
}




} // namespace ct

// kate: indent-width 4
