/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#include <stdexcept>
#include <sstream>
#include <new>

#ifdef _DEBUG
    #include <stdio.h> // for printf and fflush
#endif

#include "CxMemoryStack.h"
#include "CxDefs.h"
#include "CxOpenMpProxy.h"


namespace ct {

inline void FMemoryStack2::PushTopMark() {
#ifdef _DEBUG
   assert(m_Pos < m_Size - 8);
   *reinterpret_cast<size_t*>(&m_pData[m_Pos]) = 0xbadc0de;
   m_Pos += sizeof(size_t);
#endif
}
inline void FMemoryStack2::PopTopMark() {
#ifdef _DEBUG
   m_Pos -= sizeof(size_t);
   assert(m_Pos < m_Size - 8);
   if ( *reinterpret_cast<size_t*>(&m_pData[m_Pos]) != 0xbadc0de ) {
      printf("\n\n\n===================\nstack inconsistent!\n===================\n");
      assert(!"stack error");
   }
#endif
}

void* FMemoryStack2::Alloc( size_t nSizeInBytes )
{
   PopTopMark();
   std::size_t OldPos = m_Pos;
   m_Pos += nSizeInBytes;
   PushTopMark();
   if ( m_Pos >= m_Size )
      throw std::runtime_error("FMemoryStack2: Stack size exceeded.");
   return &m_pData[OldPos];
}

void FMemoryStack2::Free( void *p )
{
   PopTopMark();
   std::ptrdiff_t
      n = &m_pData[m_Pos] - (char*)p;
   if ( n < 0 )
      throw std::runtime_error("FMemoryStack2: Release address too low!");
   m_Pos -= n;
   PushTopMark();
}

size_t FMemoryStack2::MemoryLeft() {
   return m_Size - m_Pos;
}


FMemoryStack2::FMemoryStack2( char *pBase, std::size_t nActualSize, std::size_t nNeededSize )
   : m_pData(0)
{
   if (pBase && nActualSize >= nNeededSize)
      AssignMemory(pBase, nActualSize);
   else
      Create((nNeededSize != 0)? nNeededSize : nActualSize);
}

void FMemoryStack2::Create( std::size_t nSize )
{
  //assert_rt(m_pData == 0);
   m_Size = nSize;
   m_Pos = 0;
   m_pData = 0;
   m_bOwnMemory = false;
   if ( nSize != 0 ) {
      m_pData = new(std::nothrow) char[nSize];
      m_bOwnMemory = true;
      if ( m_pData == 0 ) {
         std::stringstream str; str.precision(2); str.setf(std::ios::fixed);
         str << "FMemoryStack2: Sorry, failed to allocate " << nSize/1e6 << " MB of memory.";
         throw std::runtime_error(str.str());
      }
      PushTopMark();
   }
}

void FMemoryStack2::AssignMemory( char *pBase_, std::size_t nSize )
{
   assert(m_pData == 0 && pBase_ != 0);
   Create(0);
   m_bOwnMemory = false;
   m_pData = pBase_;
   m_Size = nSize;
   m_Pos = 0;
   PushTopMark();
}

void FMemoryStack2::Destroy()
{
#ifdef _DEBUG
   if ( m_Size != 0 )
      PopTopMark();
   else
      assert(m_pData == 0);
   assert(m_Pos == 0);
#endif // _DEBUG
   if ( m_bOwnMemory )
      delete []m_pData;
   m_pData = 0;
}


FMemoryStack::~FMemoryStack()
{
}

FMemoryStack2::~FMemoryStack2()
{
   if ( m_bOwnMemory )
      delete []m_pData;
   m_pData = (char*)0xbadc0de;
}


void FMemoryStack::Align( uint Boundary )
{
   std::size_t
      iPos = reinterpret_cast<std::size_t>(Alloc(0)),
      iNew = ((iPos - 1) | (Boundary - 1)) + 1;
   Alloc(iNew - iPos);
}










FMemoryStackArray::FMemoryStackArray( FMemoryStack &BaseStack )
    : pBaseStack(&BaseStack)
{
    nThreads = omp_get_max_threads();
    pSubStacks = new FMemoryStack2[nThreads];
    std::size_t
        nSize = static_cast<std::size_t>(0.98 * (pBaseStack->MemoryLeft() / nThreads));

    pBaseStack->Alloc(pStackBase, nThreads * nSize);
    for ( std::size_t i = 0; i < nThreads; ++ i )
        pSubStacks[i].AssignMemory(pStackBase + i * nSize, nSize);
};

void FMemoryStackArray::Release()
{
    assert(pBaseStack != 0);
    delete []pSubStacks;
    pBaseStack->Free(pStackBase);
    pStackBase = 0;
    pBaseStack = 0;
}

FMemoryStackArray::~FMemoryStackArray()
{
    if ( pBaseStack )
        Release();
};

FMemoryStack2 &FMemoryStackArray::GetStackOfThread()
{
    int iThread = omp_get_thread_num();
    assert(iThread < (int)nThreads);
    return pSubStacks[iThread];
};






} // namespace ct

// kate: indent-width 4
