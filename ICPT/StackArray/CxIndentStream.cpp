/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#define INDENTSTREAM_IMPL
#include "CxIndentStream.h"

namespace fmt {

char const
   *const g_pSpaces = "              ";
int
   g_IndentIndex = 0,
   g_PreIndentIndex = 0;
bool
   g_IndentIndexInitialized = false;

void InitIndentStreamBuf()
{
   static int s_IndentIndex = std::ios_base::xalloc();
   g_IndentIndex = s_IndentIndex;
   // ^- explicit initizalization routine to avoid static initialization hell problems.
}



template<class FChar>
int TIndentStreamBuf<FChar>::sync() throw()
{
   long
      IndentLevel = 1;
   std::basic_string<FChar>
      *pPreIndentStr=NULL;
   if ( 0 != m_pParentStream ) {
      IndentLevel = m_pParentStream->iword(g_IndentIndex);
      void *p = m_pParentStream->pword(g_PreIndentIndex);
      pPreIndentStr = reinterpret_cast<std::basic_string<FChar>*>(p);
   }

   FBase::sync();
/*   FChar
      *pFirst = this->pbase(),
      *pLast = pFirst,
      *pEnd = this->pptr();*/
   std::string const
      &_str = this->str();
   // ^- this makes a copy. not good. Here because pgcc bypasses the
   //    output buffer and directly constructs the output string under
   //    certain circumstances (it is allowed to do that).
   FChar const
      *pFirst = &_str[0],
      *pLast = pFirst,
      *pEnd = pFirst + _str.size();
   for ( ; pLast != pEnd; pFirst = pLast ){
      // emit indentation characters, unless line is empty.
      if ( *pFirst != '\n' ) {
         if ( pPreIndentStr )
            m_Target.sputn(&*pPreIndentStr->begin(), pPreIndentStr->size());
         for ( uint i = 0; i < static_cast<uint>(IndentLevel); ++ i )
            m_Target.sputn(m_pIndentStr, m_IndentLength);
      }
      while ( (pLast != pEnd) && (*pLast != '\n') )
         ++pLast;
      if ( pLast != pEnd )
         ++pLast;
      m_Target.sputn(pFirst, pLast-pFirst);
   }
   m_Target.pubsync();

   FBase::sync();
   // clear the line buffer. Current input is not required anymore.
   this->str(m_EmptyString);
   return 0;
}

template<class FChar>
TIndentStreamBuf<FChar>::~TIndentStreamBuf() throw()
{
}

template class TIndentStream1<char>;


} // namespace fmt

// kate: space-indent on; indent-width 3; indent-mode normal;
