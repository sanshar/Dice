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

#ifndef CX_ALGEBRA_H
#define CX_ALGEBRA_H

#include <cstddef>
//#include "CxDefs.h"
//#include "CxFortranInt.h"

#ifdef _I4_
   typedef int32_t FORTINT;
#else
//    typedef boost::int64_t FORTINT;
   typedef ptrdiff_t FORTINT;
   // ^- (should default to 32bit for 32bit systems. Note that long does not have
   //     this property... it's 32bit on 64bit windows...)
#endif

typedef FORTINT const
   &FINTARG;
typedef double const
   &FDBLARG;
//#define FORT_Extern(lowercasef,UPPERCASEF) UPPERCASEF
#define FORT_Extern(lowercasef,UPPERCASEF) lowercasef

using std::ptrdiff_t;
using std::size_t;
typedef unsigned int uint;

// behold the elegance of FORTRAN77/C interfaces!
extern "C"{
        // declaration of FORTRAN77 blas/lapack routines... (to exist in ACML/MKL/etc.)

        //  C := alpha*op( A )*op( B ) + beta*C,
        // Trans*: 'N' (notranspose), 'T' (transpose) or 'C' (conjugate-transpose)(=T)
        #define DGEMM FORT_Extern(dgemm_,DGEMM)
        void DGEMM( char const &TransA, char const &TransB, FINTARG M, FINTARG N, FINTARG K,
            FDBLARG alpha, double const *A, FINTARG lda, double const *B, FINTARG ldb,
            FDBLARG beta, double *C, FINTARG ldc );

        #define DLACPY FORT_Extern(dlacpy_,DLACPY)
        void DLACPY( char const &uplo,  FINTARG M, FINTARG N,
                    double const *A, FINTARG lda, double const *B, FINTARG ldb);
  
        //  C := alpha*op(A)*op(A^T) + beta*C,
        #define DSYRK FORT_Extern(dsyrk_,DSYRK)
        void DSYRK( char const &UpLo, char const &TransA, FINTARG N, FINTARG K,
            FDBLARG alpha, double const *A, FINTARG lda,
            FDBLARG beta, double *C, FINTARG ldc );

        // y += alpha * M * x
        #define DGEMV FORT_Extern(dgemv_,DGEMV)
        void DGEMV(char const &Trans, FINTARG M, FINTARG N, FDBLARG Alpha, double const * A, FINTARG lda, double const * X, FINTARG incx, FDBLARG Beta, double * y, FINTARG incy);

        // A += alpha * x y^T
        #define DGER FORT_Extern(dger_,DGER)
        void DGER(FINTARG M, FINTARG N, double const &Alpha, double const * X, FINTARG incx, double const * Y, FINTARG incy, double * A, FINTARG ldA);

        // computes eigenvalues and eigenvectors of a symmetric matrix:
        //  jobz: 'N'/'V': compute eigenvalues only/compute also eigenvectors
        //  uplo: 'U'/'L': upper/lower triangle of A is stored.
        //  N: matrix size. A is N x N matrix.
        //  A: input matrix, output vectors. LDA: row stride A.
        //  W: output, eigenvectors in ascending order. (vector of length N).
        //  Work: work space
        //  lWork: Work space size. "For optimal efficiency, LWORK >= (NB+2)*N,
        // where NB is the blocksize for DSYTRD returned by ILAENV."
        #define DSYEV FORT_Extern(dsyev_,DSYEV)
        void DSYEV(char const &jobz, char const &uplo, FINTARG N, double *A, FINTARG LDA, double *W, double *WORK, FINTARG LWORK, FORTINT &INFO);
        #define DSYEVD FORT_Extern(dsyevd_,DSYEVD)
        void DSYEVD(char const &jobz, char const &uplo, FINTARG N, double *A, FINTARG LDA, double *W, double *WORK, FINTARG LWORK, FORTINT *IWORK, FINTARG LIWORK, FORTINT &INFO);

        #define DSYGV FORT_Extern(dsygv_,DSYGV)
        void DSYGV(FINTARG ITYPE, char const &JOBZ, char const &UPLO, FINTARG N,
            double *A, FINTARG LDA, double const *B, FINTARG LDB, double *EW,
            double *WORK, FORTINT &LWORK, FORTINT &INFO );

        // compute m x n matrix LU factorization.
        // info: =0 success. > 0: matrix is singular, factorization cannot be used
        // to solve linear systems.
        #define DGETRF FORT_Extern(dgetrf_,DGETRF)
        void DGETRF(FINTARG M, FINTARG N, double const *pA, FINTARG LDA, FORTINT *ipiv, FORTINT *INFO );
        #define DTRTRI FORT_Extern(dtrtri_,DTRTRI)
        void DTRTRI(char const &Uplo, char const &Diag, FINTARG N, double *pA, FINTARG LDA, FORTINT *info);


        // solves A * X = B for X. n: number of equations (order of A).
        // needs LU decomposition as input.
        #define DGESV FORT_Extern(dgesv_,DGESV)
        void DGESV( FINTARG n, FINTARG nrhs, double *A, FINTARG lda, FINTARG ipivot, double *B,
            FINTARG ldb, FORTINT &info );

        #define DPOTRF FORT_Extern(dpotrf_,DPOTRF)
        void DPOTRF(char const &UpLo, FINTARG n, double *A, FINTARG lda, FORTINT *info);
        #define DPOTRS FORT_Extern(dpotrs_,DPOTRS)
        void DPOTRS(char const &UpLo, FINTARG n, FINTARG nRhs, double *A, FINTARG lda, double *B, FINTARG ldb, FORTINT *info);
        #define DTRTRS FORT_Extern(dtrtrs_,DTRTRS)
        void DTRTRS(char const &UpLo, char const &Trans, char const &Diag, FINTARG N, FINTARG NRHS, double *A, FINTARG lda, double *B, FINTARG ldb, FORTINT *info);
        // ^- gna.. dtrtrs is rather useless. It just does some argument checks and
        // then calls dtrsm with side == 'L' (which probably does the same checks again).
        #define DTRSM FORT_Extern(dtrsm_,DTRSM)
        void DTRSM(char const &Side, char const &UpLo, char const &Trans, char const &Diag, FINTARG nRowsB, FINTARG nColsB, double const &Alpha, double *A, FINTARG lda, double *B, FINTARG ldb, FORTINT *info);
        #define DTRMM FORT_Extern(dtrmm_,DTRMM)
        void DTRMM(char const &Side, char const &UpLo, char const &Trans, char const &Diag, FINTARG nRowsB, FINTARG nColsB, double const &Alpha, double *A, FINTARG lda, double *B, FINTARG ldb);

        // linear least squares using divide & conquer SVD.
        #define DGELSS FORT_Extern(dgelss_,DGELSS)
        void DGELSS(FINTARG M, FINTARG N, FINTARG NRHS, double *A, FINTARG LDA, double *B, FINTARG LDB, double *S, double const &RCOND, FORTINT *RANK,
            double *WORK, FINTARG LWORK, FORTINT *INFO );

//         // compute SVD (simple driver for small cases)
//         #define DGESVD FORT_Extern(dgesvd,DGESVD)
//         void DGESVD(char const &JobU, char const &JobVT, FINTARG M, FINTARG N, double *A, FINTARG ldA, double *S, double *U, FINTARG ldU, double *Vt, FINTARG ldVt, double *Work, FINTARG lWork, FORTINT *info);
//             double *WORK, FINTARG LWORK, FORTINT *INFO );
        // divide & conquer SVD.
        #define DGESDD FORT_Extern(dgesdd_,DGESDD)
        void DGESDD(char const &JobZ, FINTARG M, FINTARG N, double *A, FINTARG ldA, double *S, double *U, FINTARG ldU, double *Vt, FINTARG ldVt, double *Work, FINTARG lWork, FORTINT *piWork, FORTINT *info);
}





namespace ct {

void Add2(double * pOut, double const * pIn, double f, size_t n);
  void Add2_0(double * pOut, double const * pIn, double f, size_t n);
  
void Mxm(double *pOut, ptrdiff_t iRowStO, ptrdiff_t iColStO,
         double const *pA, ptrdiff_t iRowStA, ptrdiff_t iColStA,
         double const *pB, ptrdiff_t iRowStB, ptrdiff_t iColStB,
         size_t nRows, size_t nLink, size_t nCols, bool AddToDest = false, double fFactor = 1.0);

// note: both H and S are overwritten. Eigenvectors go into H.
void DiagonalizeGen(double *pEw, double *pH, uint ldH, double *pS, uint ldS, uint N);
void Diagonalize(double *pEw, double *pH, uint ldH, uint N);
// note: pInAndTmp will be overwritten. Output is U: nRows x nSig, Vt : nCols x nSig, nSig = min(nRows,nCols).
void ComputeSvd(double *pU, size_t ldU, double *pSigma, double *pVt, size_t ldVt, double *pInAndTmp, size_t ldIn, size_t nRows, size_t nCols);

void MxvLame(double * pOut, ptrdiff_t iStO, double const * pMat, ptrdiff_t iRowStM, ptrdiff_t iColStM,
    double const * pIn, ptrdiff_t iStI, size_t nRows, size_t nLink, bool AddToDest = false, double fFactor = 1.0);


inline void Mxv(double * pOut, ptrdiff_t iStO, double const * pMat, ptrdiff_t iRowStM, ptrdiff_t iColStM,
    double const * pIn, ptrdiff_t iStI, size_t nRows, size_t nLink, bool AddToDest = false, double fFactor = 1.0)
{
    if (nRows <= 1 || nLink <= 1) // might not get the N/T distinction right in this case.. and mkl would make lots of errors for input 0s.
        MxvLame(pOut, iStO, pMat, iRowStM, iColStM, pIn, iStI, nRows, nLink, AddToDest, fFactor);
    else if (iRowStM == 1) {
        DGEMV('N', nRows, nLink, fFactor, pMat, iColStM,  pIn,iStI,  AddToDest? 1.0 : 0.0, pOut,iStO);
    } else {
        //assert(iColStM == 1);
        DGEMV('T', nLink, nRows, fFactor, pMat, iRowStM,  pIn,iStI,  AddToDest? 1.0 : 0.0, pOut,iStO);
    }
//     if ( iRowStM == 1 )
//         DGEMV('N', nRows, nLink, fFactor, pMat, iColStM,  pIn,iStI,  AddToDest? 1.0 : 0.0, pOut,iStO);
//     else if ( iColStM == 1 )
//         DGEMV('T', nLink, nRows, fFactor, pMat, iRowStM,  pIn,iStI,  AddToDest? 1.0 : 0.0, pOut,iStO);
//     else
//         MxvLame(pOut, iStO, pMat, iRowStM, iColStM, pIn, iStI, nRows, nLink, AddToDest, fFactor);
}


template<class FScalar>
inline FScalar Dot( FScalar const *a, FScalar const *b, size_t n )
{
    FScalar
        r = 0;
    for ( size_t i = 0; i < n; ++ i )
        r += a[i] * b[i];
    return r;
}

// r += f * x
template<class FScalar>
inline void Add( FScalar *  r, FScalar const *  x, FScalar f, size_t n )
{
    if ( f != 1.0 ) {
        for ( size_t i = 0; i < n; ++ i )
            r[i] += f * x[i];
    } else {
        for ( size_t i = 0; i < n; ++ i )
            r[i] += x[i];
    }
}

// r *= f
template<class FScalar>
inline void Scale( FScalar *r, FScalar f, size_t n )
{
    for ( size_t i = 0; i < n; ++ i )
        r[i] *= f;
}


// r = f * x
template<class FScalar>
inline void Move( FScalar *  r, FScalar const *  x, FScalar f, size_t n )
{
    for ( size_t i = 0; i < n; ++ i )
        r[i] = f * x[i];
}


} // namespace ct

#endif // CX_ALGEBRA_H

// kate: indent-width 4
