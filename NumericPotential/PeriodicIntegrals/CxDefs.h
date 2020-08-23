/* Copyright (c) 2012-2020 Gerald Knizia
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
 * along with ir/wmme (LICENSE). If not, see http://www.gnu.org/licenses/
 */

#ifndef CX_DEFS_H
#define CX_DEFS_H

#define IR_NO_THROW throw()
#define CX_NO_THROW IR_NO_THROW
#define AIC_NO_THROW IR_NO_THROW

// Define macros for restricted pointer extensions. By defining a pointer as
// ``restricted'', we promise the compiler that the pointer is not aliased in
// the current scope. E.g., that in constructs like
//
//   void fn(double *RESTRICT pOut, double const *RESTRICT pIn) {
//      pOut[0] = pIn[0];
//      pOut[1] = pIn[1];
//   },
//
// the two assignments can be scheduled out of order, because pOut[x] and pIn[x]
// can not possibly point to the same memory location.
//
// Note:
//   - RESTRICT now univerally goes behind the star ("double *RESTRICT p;").
//     That is, the *pointer* is restricted, not the data it points to.
//     (in olden tymes this was handled differently in different compilers.)
#ifdef __GNUC__ // g++ or anything which pretends to be g++ (e.g., clang, intel c++ on linux)
   #define IR_RP __restrict__
#elif defined(_MSC_VER) // microsoft c++, or anything which pretends to be msvc (e.g., intel c++ on windows)
   #define IR_RP __restrict
#elif (defined __SUNPRO_CC || defined __SUNPRO_C) // sun studio/solaris studio
    #define RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
   // compile with -restrict command line option (linux) or -Qrestrict (windows).
   #define IR_RP restrict
#else
   #define IR_RP
#endif

#ifndef RESTRICT
   #define RESTRICT IR_RP
#endif

// In some cases we need to inline driver functions as a prerequisite
// for inlining virtual functions. If the driver functions are large,
// compilers might be inclined to ignore our ``inline'' directive.
// Set up some macros to tell them that we really mean it.
#ifdef __GNUC__ // g++
   #define IR_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
   #define IR_FORCE_INLINE __forceinline
#else
   #define IR_FORCE_INLINE inline
#endif


// some other assorted defines...
#if (defined  __GNUC__ || defined __INTEL_COMPILER)
    // tell the compiler that this function will never return
    #define DECL_NORETURN __attribute__((noreturn))
    // 'insert software breakpoint here'.
    #define DEBUG_BREAK __asm__("int $0x03");
#elif defined(_MSC_VER) && !defined(__PGI)  // microsoft c++ (other win32 compilers might emulate this syntax)
    #define DECL_NORETURN __declspec(noreturn)
    //#define RESTRICT __declspec(restrict)
    #define DEBUG_BREAK __asm{ int 3 };
#elif (defined __SUNPRO_CC || defined __SUNPRO_C) // sun studio/solaris studio
    #define DECL_NORETURN __attribute((noreturn))
    #define DEBUG_BREAK __asm__("int $3");
#else
    #define DECL_NORETURN
    #define DEBUG_BREAK
#endif


// data prefetch for reading and writing.
#ifdef __GNUC__
   #define IR_PREFETCH_R(p) __builtin_prefetch(p, 0, 1)
   #define IR_PREFETCH_W(p) __builtin_prefetch(p, 1, 1)
#elif defined(_MSC_VER)
   #include <xmmintrin.h>
   #define IR_PREFETCH_R(p) _mm_prefetch((char*)p, 1)
   #define IR_PREFETCH_W(p) _mm_prefetch((char*)p, 1)
#else
   #define IR_PREFETCH_R(p) static_cast<void>(0)
   #define IR_PREFETCH_W(p) static_cast<void>(0)
#endif



// Adjust NDEBUG macro to _DEBUG macro (i.e., enable NDEBUG unless _DEBUG is set,
// and vice versa). _DEBUG is our custom "enable debug stuff" macro.
// Note:
//  - in the standard C world, NDEBUG only relates to assertions, and _DEBUG
//    (the VC standard) is unknown. I here decided to use _DEBUG as main debug
//    flag, and to adjust NDEBUG accordingly (in case someone re-includes
//    <assert.h>).
#if (defined(_DEBUG) && defined(NDEBUG))
   #undef NDEBUG
#endif

#if (!defined(_DEBUG) && !defined(NDEBUG))
   #define NDEBUG
#endif


// ask the compiler to not warn about a given symbol if it is not actually used
// (for example, because this is only a partial IR core and the symbol is used
// in some code not supplied with the partial version)
#define IR_SUPPRESS_UNUSED_WARNING(x) (void)x
#define IR_SUPPRESS_UNUSED_WARNING2(x,y) {(void)x; (void)y;}
#define IR_SUPPRESS_UNUSED_WARNING3(x,y,z) {(void)x; (void)y; (void)z;}
#define IR_SUPPRESS_UNUSED_WARNING4(x,y,z,w) {(void)x; (void)y; (void)z; (void)w;}


// re-define assert(). The main reason for doing this is that the standard
// version has a habit of hard-crashing MPI executables (e.g., Molpro), in a way
// which makes it impossible to debug them. Additionally, in some compilers
// assert()s do *not* go away, even if you do set NDEBUG.
//
// (And one can put in a debug_break into the assert fail function, which helps
// a lot if using a visual debugger like VC)
#ifdef assert
    #undef assert
#endif
#ifdef assert_rt
   #undef assert_rt
#endif
void CxAssertFail(char const *pExpr, char const *pFile, int iLine);
#define assert_rt(x) ((x)? static_cast<void>(0) : CxAssertFail(#x,__FILE__,__LINE__))

#ifdef _DEBUG
    #define assert(x) assert_rt(x)
#else
    #define assert(x) static_cast<void>(0)
#endif

#if (defined  __GNUC__ && defined __linux__ && defined(_DEBUG))
   // to allow putting "mcheck_check_all();"  (glibc's explicit heap consistency check) anywhere.
   #include <mcheck.h>
#endif


#endif // CX_DEFS_H
