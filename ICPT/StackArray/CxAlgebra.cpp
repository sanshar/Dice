/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#include <stdexcept>
#include <stdlib.h>

#include "CxAlgebra.h"
#include "CxDefs.h" // for assert.

namespace ct {

// Out = f * A * B
void Mxm(FScalar *pOut, ptrdiff_t iRowStO, ptrdiff_t iColStO,
         FScalar const *pA, ptrdiff_t iRowStA, ptrdiff_t iColStA,
         FScalar const *pB, ptrdiff_t iRowStB, ptrdiff_t iColStB,
         size_t nRows, size_t nLink, size_t nCols, bool AddToDest, FScalar fFactor )
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

    FScalar
        Beta = AddToDest? 1.0 : 0.0;
    char
        TransA, TransB;
    size_t
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

#ifdef _SINGLE_PRECISION
    sgemm_( TransA, TransB, nRows, nCols, nLink,
	    fFactor, pA, lda, pB, ldb, Beta, pOut, ldc );
#else
    dgemm_( TransA, TransB, nRows, nCols, nLink,
	    fFactor, pA, lda, pB, ldb, Beta, pOut, ldc );
#endif
}

// note: both H and S are overwritten. Eigenvectors go into H.
void DiagonalizeGen(FScalar *pEw, FScalar *pH, size_t ldH, FScalar *pS, size_t ldS, size_t N)
{
    size_t info = 0, nWork = 128*N;
    FScalar *pWork = (FScalar*)::malloc(sizeof(FScalar)*nWork);
#ifdef _SINGLE_PRECISION
    ssygv_(1, 'V', 'L', N, pH, ldH, pS, ldS, pEw, pWork, nWork, info );
#else
    dsygv_(1, 'V', 'L', N, pH, ldH, pS, ldS, pEw, pWork, nWork, info );
#endif
    ::free(pWork);
    if ( info != 0 ) throw std::runtime_error("dsygv failed.");
};

void Diagonalize(FScalar *pEw, FScalar *pH, size_t ldH, size_t N)
{
    size_t info = 0, nWork = 128*N;
    FScalar *pWork = (FScalar*)::malloc(sizeof(FScalar)*nWork);
#ifdef _SINGLE_PRECISION
    ssyev_('V', 'L', N, pH, ldH, pEw, pWork, nWork, info );
#else
    dsyev_('V', 'L', N, pH, ldH, pEw, pWork, nWork, info );
#endif
    ::free(pWork);
    if ( info != 0 ) throw std::runtime_error("dsyev failed.");
};

}

// kate: indent-width 4
