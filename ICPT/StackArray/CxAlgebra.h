/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#ifndef CX_ALGEBRA_H
#define CX_ALGEBRA_H

#include <cstddef>
#include "CxDefs.h"
#define RESTRICT AIC_RP
#include "CxFortranInt.h"

using std::ptrdiff_t;
using std::size_t;
typedef unsigned int uint;

// behold the elegance of FORTRAN77/C interfaces!
extern "C"{
        // declaration of FORTRAN77 blas/lapack routines... (to exist in ACML/MKL/etc.)

  void dcopy_(const size_t& n, double *A, const size_t& inxa, double*B, const size_t& inxb);

        //  C := alpha*op( A )*op( B ) + beta*C,
        // Trans*: 'N' (notranspose), 'T' (transpose) or 'C' (conjugate-transpose)(=T)
        void sgemm_( char const &TransA, char const &TransB, size_t &M, size_t &N, size_t &K,
            float &alpha, FScalar const *A, size_t &lda, FScalar const *B, size_t &ldb,
            float &beta, FScalar *C, size_t &ldc );

        void dgemm_( char const &TransA, char const &TransB, FINTARG M, FINTARG N, FINTARG K,
            FDBLARG alpha, double const *A, FINTARG lda, double const *B, FINTARG ldb,
            FDBLARG beta, double *C, FINTARG ldc );

        //  C := alpha*op(A)*op(A^T) + beta*C,
        #define SSYRK FORT_Extern(ssyrk,SSYRK)
        void SSYRK( char const &UpLo, char const &TransA, FINTARG N, FINTARG K,
            FDBLARG alpha, FScalar const *A, FINTARG lda,
            FDBLARG beta, FScalar *C, FINTARG ldc );

        #define SGEMV FORT_Extern(sgemv,SGEMV)
        void SGEMV(char const &Trans, FINTARG M, FINTARG N, FDBLARG Alpha, FScalar const *A, FINTARG lda, FScalar const *X, FINTARG incx, FDBLARG Beta, FScalar *y, FINTARG incy);

        #define DGEMV FORT_Extern(dgemv,DGEMV)
        void DGEMV(char const &Trans, FINTARG M, FINTARG N, FDBLARG Alpha, FScalar const *A, FINTARG lda, FScalar const *X, FINTARG incx, FDBLARG Beta, FScalar *y, FINTARG incy);

        // computes eigenvalues and eigenvectors of a symmetric matrix:
        //  jobz: 'N'/'V': compute eigenvalues only/compute also eigenvectors
        //  uplo: 'U'/'L': upper/lower triangle of A is stored.
        //  N: matrix size. A is N x N matrix.
        //  A: input matrix, output vectors. LDA: row stride A.
        //  W: output, eigenvectors in ascending order. (vector of length N).
        //  Work: work space
        //  lWork: Work space size. "For optimal efficiency, LWORK >= (NB+2)*N,
        // where NB is the blocksize for DSYTRD returned by ILAENV."
        #define SSYEV FORT_Extern(ssyev,SSYEV)
        void ssyev_(char const &jobz, char const &uplo, size_t& N, FScalar *A, size_t& LDA, FScalar *W, FScalar *WORK, size_t& LWORK, size_t &INFO );

        void dsyev_(char const &jobz, char const &uplo, size_t& N, FScalar *A, size_t& LDA, FScalar *W, FScalar *WORK, size_t& LWORK, size_t &INFO );

        #define SSYGV FORT_Extern(ssygv,SSYGV)
        void ssygv_(FINTARG ITYPE, char const &JOBZ, char const &UPLO, size_t N,
            FScalar *A, size_t LDA, FScalar const *B, size_t LDB, FScalar *EW,
            FScalar *WORK, size_t &LWORK, size_t &INFO );

        void dsygv_(size_t ITYPE, char const &JOBZ, char const &UPLO, size_t N,
            FScalar *A, size_t LDA, FScalar const *B, size_t LDB, FScalar *EW,
            FScalar *WORK, size_t &LWORK, size_t &INFO );

        // compute m x n matrix LU factorization.
        // info: =0 success. > 0: matrix is singular, factorization cannot be used
        // to solve linear systems.
        #define SGETRF FORT_Extern(sgetrf,SGETRF)
        void SGETRF(FINTARG M, FINTARG N, FScalar const *pA, FINTARG LDA, FORTINT *ipiv, FORTINT *INFO );
        #define STRTRI FORT_Extern(strtri,STRTRI)
        void STRTRI(char const &Uplo, char const &Diag, FINTARG N, FScalar *pA, FINTARG LDA, FORTINT *info);


        // solves A * X = B for X. n: number of equations (order of A).
        // needs LU decomposition as input.
        #define SGESV FORT_Extern(sgesv,SGESV)
        void SGESV( FINTARG n, FINTARG nrhs, FScalar *A, FINTARG lda, FINTARG ipivot, FScalar *B,
            FINTARG ldb, FORTINT &info );

        #define SPOTRF FORT_Extern(spotrf,SPOTRF)
        void SPOTRF(char const &UpLo, FINTARG n, FScalar *A, FINTARG lda, FORTINT *info);
        #define DPOTRS FORT_Extern(spotrs,SPOTRS)
        void SPOTRS(char const &UpLo, FINTARG n, FINTARG nRhs, FScalar *A, FINTARG lda, FScalar *B, FINTARG ldb, FORTINT *info);
        #define STRTRS FORT_Extern(strtrs,STRTRS)
        void STRTRS(char const &UpLo, char const &Trans, char const &Diag, FINTARG N, FINTARG NRHS, FScalar *A, FINTARG lda, FScalar *B, FINTARG ldb, FORTINT *info);
        // ^- gna.. dtrtrs is rather useless. It just does some argument checks and
        // then calls dtrsm with side == 'L' (which probably does the same checks again).
        #define STRSM FORT_Extern(strsm,STRSM)
        void STRSM(char const &Side, char const &UpLo, char const &Trans, char const &Diag, FINTARG nRowsB, FINTARG nColsB, FScalar const &Alpha, FScalar *A, FINTARG lda, FScalar *B, FINTARG ldb, FORTINT *info);
        #define STRMM FORT_Extern(strmm,STRMM)
        void STRMM(char const &Side, char const &UpLo, char const &Trans, char const &Diag, FINTARG nRowsB, FINTARG nColsB, FScalar const &Alpha, FScalar *A, FINTARG lda, FScalar *B, FINTARG ldb);

        // linear least squares using divide & conquer SVD.
        #define SGELSS FORT_Extern(sgelss,SGELSS)
        void SGELSS(FINTARG M, FINTARG N, FINTARG NRHS, FScalar *A, FINTARG LDA, FScalar *B, FINTARG LDB, FScalar *S, FScalar const &RCOND, FORTINT *RANK,
            FScalar *WORK, FINTARG LWORK, FORTINT *INFO );
}





namespace ct {

void Mxm(FScalar *pOut, ptrdiff_t iRowStO, ptrdiff_t iColStO,
         FScalar const *pA, ptrdiff_t iRowStA, ptrdiff_t iColStA,
         FScalar const *pB, ptrdiff_t iRowStB, ptrdiff_t iColStB,
         size_t nRows, size_t nLink, size_t nCols, bool AddToDest = false, FScalar fFactor = 1.0);

// note: both H and S are overwritten. Eigenvectors go into H.
void DiagonalizeGen(FScalar *pEw, FScalar *pH, size_t ldH, FScalar *pS, size_t ldS, size_t N);
void Diagonalize(FScalar *pEw, FScalar *pH, size_t ldH, size_t N);

inline void Mxv(FScalar *pOut, ptrdiff_t iStO, FScalar const *pMat, ptrdiff_t iRowStM, ptrdiff_t iColStM,
    FScalar const *pIn, ptrdiff_t iStI, uint nRows, uint nLink, bool AddToDest = false, FScalar fFactor = 1.0)
{
#ifdef _SINGLE_PRECISION
    if ( iRowStM == 1 )
        SGEMV('N', nRows, nLink, fFactor, pMat, iColStM,  pIn,iStI,  AddToDest? 1.0 : 0.0, pOut,iStO);
    else
        SGEMV('T', nLink, nRows, fFactor, pMat, iRowStM,  pIn,iStI,  AddToDest? 1.0 : 0.0, pOut,iStO);
#else
    if ( iRowStM == 1 )
        DGEMV('N', nRows, nLink, fFactor, pMat, iColStM,  pIn,iStI,  AddToDest? 1.0 : 0.0, pOut,iStO);
    else
        DGEMV('T', nLink, nRows, fFactor, pMat, iRowStM,  pIn,iStI,  AddToDest? 1.0 : 0.0, pOut,iStO);
#endif
}


template<class FScalar>
inline FScalar Dot( FScalar const *a, FScalar const *b, std::size_t n )
{
    FScalar
        r = 0;
    for ( std::size_t i = 0; i < n; ++ i )
        r += a[i] * b[i];
    return r;
};

// r += f * x
template<class FScalar>
inline void Add( FScalar * RESTRICT r, FScalar const * RESTRICT x, FScalar f, std::size_t n )
{
    if ( f != 1.0 ) {
        for ( std::size_t i = 0; i < n; ++ i )
            r[i] += f * x[i];
    } else {
        for ( std::size_t i = 0; i < n; ++ i )
            r[i] += x[i];
    }
};

// r *= f
template<class FScalar>
inline void Scale( FScalar *r, FScalar f, std::size_t n )
{
    for ( std::size_t i = 0; i < n; ++ i )
        r[i] *= f;
};


// r = f * x
template<class FScalar>
inline void Move( FScalar * RESTRICT r, FScalar const * RESTRICT x, FScalar f, std::size_t n )
{
    for ( std::size_t i = 0; i < n; ++ i )
        r[i] = f * x[i];
};


} // namespace ct

#endif // CX_ALGEBRA_H

// kate: indent-width 4
