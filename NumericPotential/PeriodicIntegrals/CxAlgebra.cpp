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

#include <algorithm> // for std::min
#include <stdexcept>
#include <stdlib.h>

#include "CxAlgebra.h"
#include "CxDefs.h" // for assert.

namespace ct {

void Add2(double * pOut, double const * pIn, double f, size_t n)
{
  size_t i = 0;
  for ( ; i < (n & ~3); i += 4 ) {
    pOut[i]   += f * pIn[i];
    pOut[i+1] += f * pIn[i+1];
    pOut[i+2] += f * pIn[i+2];
    pOut[i+3] += f * pIn[i+3];
  }
  pOut += i;
  pIn += i;
  switch(n - i) {
    case 3: pOut[2] += f*pIn[2];
    case 2: pOut[1] += f*pIn[1];
    case 1: pOut[0] += f*pIn[0];
    default: break;
  }
}

  
// Out = f * A * B
void Mxm(double *pOut, ptrdiff_t iRowStO, ptrdiff_t iColStO,
         double const *pA, ptrdiff_t iRowStA, ptrdiff_t iColStA,
         double const *pB, ptrdiff_t iRowStB, ptrdiff_t iColStB,
         size_t nRows, size_t nLink, size_t nCols, bool AddToDest, double fFactor )
{
    assert( iRowStO == 1 || iColStO == 1 );
    assert( iRowStA == 1 || iColStA == 1 );
    assert( iRowStB == 1 || iColStB == 1 );
    // ^- otherwise dgemm directly not applicable. Would need local copy
    // of matrix/matrices with compressed strides.

//     if ( nRows == 1 || nLink == 1 || nCols == 1 ) {
//         if ( !AddToDest )
//             for ( uint ic = 0; ic < nCols; ++ ic )
//                 for ( uint ir = 0; ir < nRows; ++ ir )
//                     pOut[ir*iRowStO + ic*iColStO] = 0;
//
//         for ( uint ic = 0; ic < nCols; ++ ic )
//             for ( uint ir = 0; ir < nRows; ++ ir )
//                 for ( uint il = 0; il < nLink; ++ il )
//                     pOut[ir*iRowStO + ic*iColStO] += fFactor * pA[ir*iRowStA + il*iColStA] * pB[il*iRowStB + ic*iColStB];
//         return;
//     }

    double
        Beta = AddToDest? 1.0 : 0.0;
    char
        TransA, TransB;
    FORTINT
        lda, ldb,
        ldc = (iRowStO == 1)? iColStO : iRowStO;

    if ( iRowStA == 1 ) {
        TransA = 'N'; lda = iColStA;
    } else {
        TransA = 'T'; lda = iRowStA;
    }
    if ( iRowStB == 1 ) {
        TransB = 'N'; ldb = iColStB;
    } else {
        TransB = 'T'; ldb = iRowStB;
    }

    DGEMM( TransA, TransB, nRows, nCols, nLink,
        fFactor, pA, lda, pB, ldb, Beta, pOut, ldc );
}

//// this one is used if neither the column nor the row stride in pMat is unity.
//void MxvLame(double *RESTRICT pOut, ptrdiff_t iStO, double const *RESTRICT pMat, ptrdiff_t iRowStM, ptrdiff_t iColStM,
//    double const *RESTRICT pIn, ptrdiff_t iStI, size_t nRows, size_t nLink, bool AddToDest, double fFactor)
//{
//    for (size_t iRow = 0; iRow < nRows; ++ iRow) {
//        double
//            d = 0;
//        double const
//            *RESTRICT pM = &pMat[iRowStM * iRow];
//        for (size_t iLink = 0; iLink < nLink; ++ iLink) {
//            d += pIn[iStI * iLink] * pM[iColStM * iLink];
//        }
//        d *= fFactor;
//
//        double *RESTRICT r = &pOut[iStO * iRow];
//        if (AddToDest)
//            *r += d;
//        else
//            *r = d;
//    }
//}

// this one is used if neither the column nor the row stride in pMat is unity.
void MxvLameG(double *RESTRICT pOut, ptrdiff_t iStO, double const *RESTRICT pMat, ptrdiff_t iRowStM, ptrdiff_t iColStM,
   double const *RESTRICT pIn, ptrdiff_t iStI, size_t nRows, size_t nLink, bool AddToDest, double fFactor)
{
   for (size_t iRow = 0; iRow < nRows; ++iRow) {
      double
         d = 0;
      double const
         *RESTRICT pM = &pMat[iRowStM * iRow];
      for (size_t iLink = 0; iLink < nLink; ++iLink) {
         d += pIn[iStI * iLink] * pM[iColStM * iLink];
      }
      d *= fFactor;

      double *RESTRICT r = &pOut[iStO * iRow];
      if (AddToDest)
         *r += d;
      else
         *r = d;
   }
}

// this one is used if neither the column nor the row stride in pMat is unity.
void MxvLame(double *RESTRICT pOut, ptrdiff_t iStO, double const *RESTRICT pMat, ptrdiff_t iRowStM, ptrdiff_t iColStM,
   double const *RESTRICT pIn, ptrdiff_t iStI, size_t nRows, size_t nLink, bool AddToDest, double fFactor)
{
   if (iStO != 1 || iStI != 1)
      return MxvLameG(pOut, iStO, pMat, iRowStM, iColStM, pIn, iStI, nRows, nLink, AddToDest, fFactor);
   for (size_t iRow = 0; iRow < nRows; ++iRow) {
      double
         d = 0;
      double const
         *RESTRICT pM = &pMat[iRowStM * iRow];
      if (iColStM == 1) {
         for (size_t iLink = 0; iLink < nLink; ++iLink) {
            d += pIn[iLink] * pM[iLink];
         }
      } else {
         for (size_t iLink = 0; iLink < nLink; ++iLink) {
            d += pIn[iLink] * pM[iColStM * iLink];
         }
      }
      d *= fFactor;

      double *RESTRICT r = &pOut[iRow];
      if (AddToDest)
         *r += d;
      else
         *r = d;
   }
}



// note: both H and S are overwritten. Eigenvectors go into H.
void DiagonalizeGen(double *pEw, double *pH, uint ldH, double *pS, uint ldS, uint N)
{
    FORTINT info = 0, nWork = 128*N;
    double *pWork = (double*)::malloc(sizeof(double)*nWork);
    DSYGV(1, 'V', 'L', N, pH, ldH, pS, ldS, pEw, pWork, nWork, info );
    ::free(pWork);
    if ( info != 0 ) throw std::runtime_error("dsygv failed.");
}

// void Diagonalize(double *pEw, double *pH, uint ldH, uint N)
// {
//     FORTINT info = 0, nWork = 128*N;
//     double *pWork = (double*)::malloc(sizeof(double)*nWork);
//     DSYEV('V', 'L', N, pH, ldH, pEw, pWork, nWork, info );
//     ::free(pWork);
//     if ( info != 0 ) throw std::runtime_error("dsyev failed.");
// }

void Diagonalize(double *pEw, double *pH, uint ldH, uint N)
{
    if (N == 0)
        return;
    FORTINT info = 0;
    // workspace query.
    double fWork = 0;
    FORTINT nWork, niWork[2] = {0};
    DSYEVD('V', 'L', N, pH, ldH, pEw, &fWork, -1, &niWork[0], -1, info);
    if ( info != 0 ) throw std::runtime_error("dsyevd workspace query failed.");
    nWork = FORTINT(fWork);

    double *pWork = (double*)::malloc(sizeof(double)*nWork+1000);
    FORTINT *piWork = (FORTINT*)::malloc(sizeof(FORTINT)*niWork[0]+1000);
    DSYEVD('V', 'L', N, pH, ldH, pEw, pWork, nWork, piWork, niWork[0], info);
    ::free(piWork);
    ::free(pWork);
    if ( info != 0 ) throw std::runtime_error("dsyevd failed.");

//     if ( info != 0 ) {
//         std::stringstream str;
//         str << "Something went wrong when trying to diagonalize a " << InOut.nRows << "x" << InOut.nCols << " matrix. "
//             << "DSYEVD returned error code " << info << ".";
//         throw std::runtime_error(str.str());
//     }
}


// U: nRows x nSig, Vt: nCols x nSig,
// where nSig = min(nRows, nCols)
void ComputeSvd(double *pU, size_t ldU, double *pSigma, double *pVt, size_t ldVt, double *pInAndTmp, size_t ldIn, size_t nRows, size_t nCols)
{
    size_t
        nSig = std::min(nRows, nCols);
    assert(ldU >= nRows && ldVt >= nSig && ldIn >= nRows);
    FORTINT
        lWork = 0,
        info = 0;
    FORTINT
        *piWork = (FORTINT*)::malloc(sizeof(FORTINT)*8*nSig);
    // workspace query.
    double
        flWork = 0;
    DGESDD('S', nRows, nCols, pInAndTmp, ldIn, pSigma, pU, ldU, pVt, ldVt, &flWork, -1, piWork, &info);
    if ( info != 0 ) {
        ::free(piWork); // my understanding of the docs is that piWork needs to be valid for the workspace query...
        throw std::runtime_error("dgesdd workspace query failed.");
    }
    lWork = FORTINT(flWork);
    double
        *pWork = (double*)::malloc(sizeof(double)*lWork);
    DGESDD('S', nRows, nCols, pInAndTmp, ldIn, pSigma, pU, ldU, pVt, ldVt, pWork, lWork, piWork, &info);
    ::free(pWork);
    ::free(piWork);
    if ( info != 0 )
        throw std::runtime_error("dgesdd failed.");
}



}

// kate: indent-width 4
