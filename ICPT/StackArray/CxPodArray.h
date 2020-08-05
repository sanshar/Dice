/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#ifndef CX_PODARRAY_H
#define CX_PODARRAY_H

#include <string.h> // for memcpy
#include <stdlib.h> // for malloc/free
#include "iostream"
#include "CxDefs.h"
namespace ct {

// swap two primitive values. Here such that we need not include <algorithm> here.
template<class FType>
void swap1(FType &A, FType &B){
   FType t = A; A = B; B = t;
}


// A dynamic array of POD (``plain old data'') types that can be
// copied via a memcpy. Main point for this is that std::vector does not
// allow containing C-style arrays (e.g., double [3]), because C-style
// arrays are not assignable. Additionally, std::vector can be *very* slow
// when allocating large amounts of data, because that data is set to
// zero on resize. This class explicitly does not do that: It has RAII
// semantics, but non-explicitly touched data is just random.
//
// It is effectively a 'buffer-ptr + size' pair.
template<class FType>
struct TArray
{
   typedef FType *iterator;
   typedef FType const *const_iterator;
   typedef ::size_t size_type;

   iterator begin() { return m_pData; }
   iterator end() { return m_pDataEnd; }
   const_iterator begin() const { return m_pData; }
   const_iterator end() const { return m_pDataEnd; }

   FType &front() { return *m_pData; }
   FType &back() { return *(m_pDataEnd-1); }
   FType const &front() const { return *m_pData; }
   FType const &back() const { return *(m_pDataEnd-1); }

   FType &operator[] (size_type i) { return m_pData[i]; }
   FType const &operator[] (size_type i) const  { return m_pData[i]; }

   size_type size() const { return m_pDataEnd - m_pData; }
   bool empty() const { return m_pData == m_pDataEnd; }

   // WARNING: contrary to std::vector, this function *DOES NOT*
   // initialize (or touch, for that matter) the newly created data.
   // If you want that behavior, use the other resize function.
   void resize(size_type n) {
      reserve(n);
      m_pDataEnd = m_pData + n;
   };

   void resize(size_type n, FType t) {
      size_type old_size = size();
      resize(n);
      if ( old_size < n ) {
         for ( size_type i = old_size; i < n; ++ i )
            m_pData[i] = t;
      }
   };

   void clear() {
      ::free(m_pData);
      m_pData = 0;
      m_pDataEnd = 0;
      m_pReservedEnd = 0;
   };

   // memset the entire array to 0.
   void clear_data() {
      memset(m_pData, 0, sizeof(FType)*size());
   };

   void resize_and_clear(size_type n) {
      resize(n);
      clear_data();
   }

   void push_back( FType const &t ) {
      if ( size() + 1 > static_cast<size_type>(m_pReservedEnd - m_pData) ) {
         reserve(2 * size() + 1);
      }
      m_pDataEnd[0] = t;
      ++m_pDataEnd;
   };

   void pop_back() {
      assert(!empty());
      m_pDataEnd -= 1;
   }

   void reserve(size_type n) {
      if ( static_cast<size_type>(m_pReservedEnd - m_pData) < n ) {
         FType *pNewData = static_cast<FType*>(::malloc(sizeof(FType) * n));
         size_type
            nSize = size();
         if ( nSize != 0 )
            ::memcpy(pNewData, m_pData, sizeof(FType) * nSize);
         ::free(m_pData);
         m_pData = pNewData;
         m_pDataEnd = m_pData + nSize;
         m_pReservedEnd = m_pData + n;
      }
   };


   TArray()
      : m_pData(0), m_pDataEnd(0), m_pReservedEnd(0)
   {}

   TArray(TArray const &other)
      : m_pData(0), m_pDataEnd(0), m_pReservedEnd(0)
   {
      *this = other;
   };

   // WARNING: array's content not initialized with this function!
   // (with intention!)
   explicit TArray(size_type n)
      : m_pData(0), m_pDataEnd(0), m_pReservedEnd(0)
   {
      resize(n);
   }

   TArray(size_type n, FType t)
      : m_pData(0), m_pDataEnd(0), m_pReservedEnd(0)
   {
      resize(n);
      for ( size_type i = 0; i < n; ++ i )
         m_pData[i] = t;
   }

   ~TArray() {
      ::free(m_pData);
      m_pData = 0;
   }

   void operator = (TArray const &other) {
      resize(other.size());
      memcpy(m_pData, other.m_pData, sizeof(FType) * size());
   };

   void swap(TArray &other) {
      swap1(m_pData, other.m_pData);
      swap1(m_pDataEnd, other.m_pDataEnd);
      swap1(m_pReservedEnd, other.m_pReservedEnd);
   };

   template<class FRandomIt>
   void assign(FRandomIt begin, FRandomIt end){
      resize(end - begin);
      for ( size_type i = 0; i < size(); ++ i )
         m_pData[i] = begin[i];
   }
private:
   FType
      // start of controlled array. may be 0 if no data is contained.
      *m_pData,
      // end of actual data
      *m_pDataEnd,
      // end of reserved space (i.e., of the space *this owns)
      *m_pReservedEnd;
};

} // namespace ct

namespace std {
   template<class FType>
   void swap( ct::TArray<FType> &A, ct::TArray<FType> &B ) {
      A.swap(B);
   }
}


#endif // CX_PODARRAY_H
