/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#ifndef CX_INDENTSTREAM_H
#define CX_INDENTSTREAM_H

#include <ostream>
#include <sstream>
#include <string>
#include "assert.h"

typedef unsigned int
   uint;

namespace fmt {
   extern int
      g_IndentIndex,
      g_PreIndentIndex;
   extern bool
      g_IndentIndexInitialized;

   /// stream modifier which changes the indendation level on an
   /// ostream. Has effect only if the ostream/ostream adapter
   /// supports this functionality.
   struct ind{
      ind( int IndentDelta = +1, bool endl = false )
         : nIndentDelta(IndentDelta), EmitEndl(endl)
      {}
      int nIndentDelta;
      bool EmitEndl;
   };
   struct unind : public ind { unind() : ind(-1,false) {}; };
   struct eind : public ind { eind() : ind(+1,true) {}; };
   struct eunind : public ind { eunind() : ind(-1,true) {}; };


   template<class FChar>
   std::basic_ostream<FChar> &operator << ( std::basic_ostream<FChar> &out, ind const &in )
   {
      if ( in.EmitEndl )
         out << std::endl;
      else
         out.flush();
      long
         &nIndentLevel = out.iword(g_IndentIndex);
      nIndentLevel += in.nIndentDelta;
      return out;
   }

   // a pre-indent is a string which is emitted at the start of each line, before
   // the spaces for the indendation take effect. This allows emitting some left-
   // aligned data before the structured data. It can be used, for example, to
   // implement line numbers or similar debug information.
   template<class FChar>
   struct set_preindent {
      // If used, pre-indent for the stream will be taken from the referenced
      // string; if set to 0, pre-indent is dsiabled. Note that this sets a
      // *pointer*. MAKE SURE that the corresponding string object is not
      // destroyed until pre-indent is reset to 0! Otherwise crash! (in the
      // best case!)
      set_preindent(std::basic_string<FChar> *p)
         : pString(p)
      {};
      std::basic_string<FChar>
         *pString;
   };

   template<class FChar>
   std::basic_ostream<FChar> &operator << ( std::basic_ostream<FChar> &out, set_preindent<FChar> const &in )
   {
      out.flush();
      void
         *&pStrElm = out.pword(g_PreIndentIndex);
      pStrElm = static_cast<void*>(in.pString);
      return out;
   }
}

#ifdef INDENTSTREAM_IMPL

namespace fmt {

// stream_buf which allows appending prefixes at the beginning of
// any new line.
// It does so by caching a line each, which is transmitted to the
// target stream_buf when full.
template<class FChar>
class TIndentStreamBuf : public std::basic_stringbuf<FChar>
{
public:
   typedef std::basic_stringbuf<FChar>
      FBase;

   TIndentStreamBuf( std::basic_streambuf<FChar> &Target,
            uint nIndentLength, FChar const *pIndentStr,
            std::basic_ostream<FChar> *pParentStream )
      : FBase(std::ios_base::out), m_Target(Target), m_pParentStream(pParentStream),
        m_pIndentStr(pIndentStr), m_IndentLength(nIndentLength)
   {
      if (!g_IndentIndexInitialized) {
         g_IndentIndex = std::ios_base::xalloc();
         g_PreIndentIndex = std::ios_base::xalloc();
         g_IndentIndexInitialized = true;
      }
   };

   ~TIndentStreamBuf() throw();
protected:
   int sync() throw(); // override
private:
   std::basic_streambuf<FChar>
      &m_Target;
   std::basic_ostream<FChar>
      *m_pParentStream;
   std::basic_string<FChar>
      m_EmptyString;
   FChar const
      *m_pIndentStr;
   int
      m_IndentLength;
};

extern char const
   *const g_pSpaces;

/// ostream adapter class adding support for indentation to another
/// ostream object. Indendation level is either modified using ind()
/// objects (UseStreamModifiers==true) of by building TIndentOstream
/// cascades (less efficient, UseStreamModifiers==false).
template<class FChar>
class TIndentOstream : public std::basic_ostream<FChar>
{
public:
   typedef std::basic_ostream<FChar>
      FBase;
   /// UseStreamModifiers: if true, use reind(), ind() and unind()
   TIndentOstream( TIndentStreamBuf<FChar> &Buf )
      : FBase( &Buf )
   {};
};

// holds a indent-streambuf and a indent-stream object.
template<class FChar>
class TIndentStream1
{
   TIndentStreamBuf<FChar>
      m_IndentBuf;
public:
   typedef std::basic_ostream<FChar>
      FBase;
   TIndentOstream<FChar>
      stream;
   TIndentStream1( FBase &Target, bool UseStreamModifiers = false,
         int IndentWidth = 3, FChar const *pIndentStr = g_pSpaces )
      : m_IndentBuf( *Target.rdbuf(), IndentWidth, pIndentStr, UseStreamModifiers? &this->stream : 0 ),
        stream( m_IndentBuf )
   {};
};

typedef TIndentStream1<char>
   FIndentStream1;

} // namespace fmt

#endif // INDENTSTREAM_IMPL


#endif // CX_INDENTSTREAM_H

// kate: space-indent on; indent-width 3; indent-mode normal;
