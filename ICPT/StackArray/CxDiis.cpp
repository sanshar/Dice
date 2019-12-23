/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cmath>

#include "CxTypes.h"
#include "CxAlgebra.h"
#include "CxDiis.h"
#include "CxMemoryStack.h"

namespace ct {
uint const VecNotPresent = 0xffff;


void PrintMatrixGen( std::ostream &out, double const *pData,
        uint nRows, uint nRowStride, uint nCols, uint nColStride,
        std::string const &Name );

  //typedef FDiisTarget::FScalar
  //FScalar;

void ErrorExit1(std::string const &s) {
    throw std::runtime_error(s);
}

void LinearSolveSymSvd(FScalar *pOut, FScalar *pMat, uint nStr, FScalar *pIn, uint nDim, FScalar Thr, FMemoryStack &Mem)
{
    FScalar
        *pEvs = Mem.AllocN(nDim * nDim, pMat[0]),
        *pEws = Mem.AllocN(nDim, pMat[0]),
        *pXv = Mem.AllocN(nDim, pMat[0]); // input vectors in EV basis.
    for (uint j = 0; j < nDim; ++ j)
        for (uint i = 0; i < nDim; ++ i)
            pEvs[i + nDim*j] = -pMat[i + nStr*j]; // <- beware of minus.
    Diagonalize(pEws, pEvs, nDim, nDim);

    Mxv(pXv,1, pEvs,nDim,1,  pIn,1, nDim,nDim);
    for (uint iEw = 0; iEw != nDim; ++ iEw)
        if (std::abs(pEws[iEw]) >= Thr)
            pXv[iEw] /= -pEws[iEw];
            // ^- note that this only screens by absolute value.
            // no positive semi-definiteness is assumed!
        else
            pXv[iEw] = 0.;
    Mxv(pOut,1, pEvs,1,nDim, pXv,1, nDim,nDim);

    Mem.Free(pEvs);
}


FDiisState::FDiisState( FMemoryStack &Memory, FDiisOptions const &Options,
        FStorageBlock const &Storage )
    : m_Options( Options ),
      m_Storage( Storage ),
      m_Memory(Memory),
      m_ErrorMatrix( Options.nMaxDim, Options.nMaxDim )
{
    m_nAmplitudeLength = iNotPresent;
    m_nResidualLength = iNotPresent;

    if ( m_Storage.pDevice == 0 ) {
        // no external device provided. Use an internal one
        // which just keeps resident memory buffers.
        m_Storage = FStorageBlock(m_StorageDeviceToUseIfNoneProvided,
            m_StorageDeviceToUseIfNoneProvided.AllocNewRecord());
    }

    if ( nMaxDim() >= nMaxDiisCapacity )
        ErrorExit1( "Trying to construct a DIIS object with higher number of"
            " subspace vectors than currently allowed. Increase nMaxDiisCapacity." );

    Reset();
};

void FDiisState::Reset()
{
//     m_nDim = 0;
    for ( uint i = 0; i < nMaxDiisCapacity; ++ i )
        m_iVectorAge[i] = VecNotPresent;
    m_iNext = 0;
};


FStorageDevice::FRecord FDiisState::ResidualRecord( uint iVec )
{
    // store all residuals at the front of the record.
    assert( m_nResidualLength != iNotPresent );
    FStorageDevice::FRecord
        r(m_Storage.Record);
    r.BaseOffset += m_nResidualLength * iVec;
    return r;
};

FStorageDevice::FRecord FDiisState::AmplitudeRecord( uint iVec )
{
    // store amplitudes behind the residuals.
    assert( m_nAmplitudeLength != iNotPresent );
    FStorageDevice::FRecord
        r(m_Storage.Record);
    r.BaseOffset += m_nResidualLength * nMaxDim();
    r.BaseOffset += m_nAmplitudeLength * iVec;
    return r;
};


struct FDiisTargetLock
{
    FDiisTargetLock( FDiisTarget &TR_ ) : TR(TR_) {
        TR.Prepare();
    };
    ~FDiisTargetLock(){
        TR.Finish();
    };
    FDiisTarget &TR;
};

uint FDiisState::nLastDim() const
{
    uint nDim = 0;
    for ( uint i = 0; i < nMaxDim(); ++ i )
        if ( m_iVectorAge[i] < VecNotPresent )
            nDim += 1;
    return nDim;
}

void FDiisState::FindUsefulVectors(uint *iUsedVecs, uint &nDim, FScalar &fBaseScale, uint iThis)
{
    // remove lines from the system which correspond to vectors which are too bad
    // to be really useful for extrapolation purposes, but which might break
    // numerical stability if left in.
    FScalar const
        fThrBadResidual = 1e12;
    FScalar
        fBestResidualDot = m_ErrorMatrix(iThis,iThis),
        fWorstResidualDot = fBestResidualDot;
    assert(m_iVectorAge[iThis] < VecNotPresent);
    for ( uint i = 0; i < nMaxDim(); ++ i ) {
        if ( m_iVectorAge[i] >= VecNotPresent ) continue;
        fBestResidualDot = std::min( m_ErrorMatrix(i,i), fBestResidualDot );
    }
    nDim = 0;
    for ( uint i = 0; i < nMaxDim(); ++ i ){
        if ( i != iThis && m_iVectorAge[i] >= VecNotPresent )
            continue;
        if ( i != iThis && m_ErrorMatrix(i,i) > fBestResidualDot * fThrBadResidual) {
            m_iVectorAge[i] = VecNotPresent; // ignore this slot next time.
            continue;
        }
        fWorstResidualDot = std::max( m_ErrorMatrix(i,i), fWorstResidualDot );
        iUsedVecs[nDim] = i;
        ++ nDim;
    };

    fBaseScale = std::sqrt(fWorstResidualDot * fBestResidualDot);
    if ( fBaseScale <= 0. )
        fBaseScale = 1.;
};

void FDiisState::PerformDiis( FDiisTarget &TR, FScalar W1 )
{
    using std::abs;

    if ( nMaxDim() <= 1 )
        // DIIS has been disabled.
        return;

    // see if we have enough memory left for diis matrices and
    // for a reasonable file buffer in FileAndMemOp.
    if ( m_Memory.MemoryLeft() < sizeof(FScalar) * (25000 + 1338) )
        ErrorExit1("DIIS: Not enough memory left to perform DIIS.");

    FDiisTargetLock
        Lock(TR);
    void
        *pTopOfStack = m_Memory.Alloc(0);

    if ( m_nAmplitudeLength == iNotPresent ) {
        // vector not yet initialized. vector length was not known previously.
        m_nAmplitudeLength = TR.nAmplitudeLength();
        m_nResidualLength = TR.nResidualLength();

        // reserve memory for iterative subspace residuals/amplitudes on target record.
        FStreamSize
            TotalLength = nMaxDim() * ( m_nAmplitudeLength + m_nResidualLength );
        m_Storage.pDevice->Reserve( m_Storage.Record, TotalLength );
        m_Weights.resize( nMaxDim() );
    } else {
        if ( m_nAmplitudeLength != TR.nAmplitudeLength() ||
            m_nResidualLength != TR.nResidualLength() )
            ErrorExit1("DIIS: Lengths of different amplitude/residual vectors differ.");
    }

    FScalar
        fThisResidualDot = TR.OwnResidualDot();
    m_LastResidualNormSq = fThisResidualDot;

    if ( m_iNext == 0 && fThisResidualDot > m_Options.ResidualThresh )
        // current vector is to be considered too wrong to be useful for DIIS
        // purposes. Don't store it.
        return;

    uint
        iThis = m_iNext;

    assert(iThis < nMaxDim() && nMaxDim() < nMaxDiisCapacity);
    m_ErrorMatrix(iThis,iThis) = fThisResidualDot;
    for ( uint i = 0; i < nMaxDim(); ++ i )
        m_iVectorAge[i] += 1;
    m_iVectorAge[iThis] = 0;

    // find set of vectors actually used in the current run and
    // find their common size scale.
    uint
        iUsedVecs[ nMaxDiisCapacity + 1 ],
        nDim;
        // ^- note: this is the DIIS dimension--the actual matrices and vectors have
        // dimension nDim+1 due to the Lagrange-Multipliers!
    FScalar
        fBaseScale;
    FindUsefulVectors(&iUsedVecs[0], nDim, fBaseScale, iThis);
    // transform iThis into a relative index.
    for ( uint i = 0; i < nDim; ++ i )
        if ( iThis == iUsedVecs[i] ) {
            iThis = i;
            break;
        }

    // make array of sub-records describing all present subspace vectors
    FStorageDevice::FRecord
        ResRecs[ nMaxDiisCapacity + 1 ],
        AmpRecs[ nMaxDiisCapacity + 1 ];
    assert( nDim < nMaxDiisCapacity );
    for ( uint i = 0; i < nDim; ++ i ){
        ResRecs[i] = ResidualRecord(iUsedVecs[i]);
        AmpRecs[i] = AmplitudeRecord(iUsedVecs[i]);
    }

    // write current amplitude and residual vectors to their designated place
    TR.SerializeTo( ResRecs[iThis], AmpRecs[iThis], *m_Storage.pDevice, m_Memory );
    m_Weights[iUsedVecs[iThis]] = W1;

    // go through previous residual vectors and form the dot products with them
    FDiisVector
        ResDot(nDim);
    TR.CalcResidualDots( ResDot.data(), ResRecs, nDim, iThis,
            *m_Storage.pDevice, m_Memory );
    ResDot[iThis] = fThisResidualDot;

    // update resident error matrix with new residual-dots
    for ( uint i = 0; i < nDim; ++ i ) {
        m_ErrorMatrix(iUsedVecs[i], iUsedVecs[iThis]) = ResDot[i];
        m_ErrorMatrix(iUsedVecs[iThis], iUsedVecs[i]) = ResDot[i];
    }

    // build actual DIIS system for the subspace used.
    FDiisVector
        Rhs(nDim+1),
        Coeffs(nDim+1);
    FDiisMatrix
        B(nDim+1, nDim+1);

    // Factor out common size scales from the residual dots.
    // This is done to increase numerical stability for the case when _all_
    // residuals are very small.
    for ( uint nRow = 0; nRow < nDim; ++ nRow )
        for ( uint nCol = 0; nCol < nDim; ++ nCol )
            B(nRow, nCol) = m_ErrorMatrix(iUsedVecs[nRow], iUsedVecs[nCol])/fBaseScale;

    // make Lagrange/constraint lines.
    for ( uint i = 0; i < nDim; ++ i ) {
        B(i, nDim) = -m_Weights[iUsedVecs[i]];
        B(nDim, i) = -m_Weights[iUsedVecs[i]];
        Rhs[i] = 0.0;
    }
    B(nDim, nDim) = 0.0;
    Rhs[nDim] = -W1;

    // invert the system, determine extrapolation coefficients.
    LinearSolveSymSvd( Coeffs.data(), B.data(), nDim+1, Rhs.data(), nDim+1, 1.0e-10, m_Memory );

    // Find a storage place for the vector in the next round. Either
    // an empty slot or the oldest vector.
    uint iOldestAge = m_iVectorAge[0];
    m_iNext = 0;
    for ( uint i = nMaxDim(); i != 0; -- i ){
        if ( iOldestAge <= m_iVectorAge[i-1] ) {
            iOldestAge = m_iVectorAge[i-1];
            m_iNext = i-1;
        }
    }

//     bool
//         PrintDiisState = true;
//     if ( PrintDiisState ) {
//         std::ostream &xout = std::cout;
//         xout << "  iUsedVecs: "; for ( uint i = 0; i < nDim; ++ i ) xout << " " << iUsedVecs[i]; xout << std::endl;
//         PrintMatrixGen( xout, m_ErrorMatrix.data(), nMaxDim(), 1, nMaxDim(), m_ErrorMatrix.nStride(), "DIIS-B (resident)" );
//         PrintMatrixGen( xout, B.data(), nDim+1, 1, nDim+1, B.nStride(), "DIIS-B/I" );
//         PrintMatrixGen( xout, Rhs.data(), 1, 1, nDim+1, 1, "DIIS-Rhs" );
//         PrintMatrixGen( xout, Coeffs.data(), 1, 1, nDim+1, 1, "DIIS-C" );
//         xout << std::endl;
//     }

    // now actually perform the extrapolation on the residuals
    // and amplitudes.
    m_LastAmplitudeCoeff = Coeffs[iThis];
    TR.InterpolateFrom( Coeffs[iThis], Coeffs.data(), ResRecs, AmpRecs,
        nDim, iThis, *m_Storage.pDevice, m_Memory );

    // done.
//     assert_rt( m_Memory.IsOnTop(pTopOfStack) );
    m_Memory.Free(pTopOfStack);
};

// convenience function to run DIIS on a block set in memory.
// pOth: set of amplitudes to extrapolate, but not to include in the
// DIIS system itself.
void FDiisState::operator() ( FScalar *pAmp, size_t nAmpSize, FScalar *pRes,
    size_t nResSize, FScalar *pOth, size_t nOthSize )
{
    if ( nResSize == static_cast<size_t>(-1) )
        nResSize = nAmpSize;
    FDiisTargetMemoryBlockSet
        TR(pAmp, nAmpSize, pRes, nResSize);
    if ( pOth != 0 )
        TR.AddAmpBlock(pOth, nOthSize);
    PerformDiis(TR);
};


FDiisTarget::~FDiisTarget()
{
};

void FDiisTarget::Prepare()
{
};

void FDiisTarget::Finish()
{
};


FDiisTargetMemoryBlockSet::~FDiisTargetMemoryBlockSet()
{
};

FStreamSize FDiisTargetMemoryBlockSet::nBlockSetLength( FMemoryBlockSet const &BlockSet )
{
    FStreamSize
        r = 0;
    FMemoryBlockSet::const_iterator
        itBlock;
    _for_each( itBlock, BlockSet )
        r += itBlock->nSize;
    return sizeof(FScalar) * r;
}

FStreamSize FDiisTargetMemoryBlockSet::nResidualLength(){
    return nBlockSetLength( m_ResBlocks );
};

FStreamSize FDiisTargetMemoryBlockSet::nAmplitudeLength(){
    return nBlockSetLength( m_AmpBlocks );
};

FScalar FDiisTargetMemoryBlockSet::OwnResidualDot()
{
    FScalar
        r = 0;
    FMemoryBlockSet::const_iterator
        itBlock;
    _for_each( itBlock, m_ResBlocks )
        r += Dot( itBlock->pData, itBlock->pData, itBlock->nSize );
    return r;
}

template<class FScalar>
void FileAndMemOp( FFileAndMemOp Op, FScalar &f, FMemoryBlock *pMemBlocks,
    std::size_t nMemBlocks, FStorageBlock const &Rec_,
    FMemoryStack &Temp )
{
    FStorageBlock
        Rec( Rec_ );
    FScalar
        f2 = 0.0;
    for ( uint iBlock = 0; iBlock < nMemBlocks; ++ iBlock ){
        //xout << "FM-OP: " << Op << "  " << Rec.Record.iRecord << fmt::ff(f,14,6) << " off " << fmt::fi(Rec.Record.BaseOffset,10) << " len" << fmt::fi(sizeof(FScalar) * pMemBlocks->nSize,10) << std::endl;
        if ( Op == OP_WriteFile ) {
            Rec.pDevice->Write( Rec.Record, pMemBlocks->pData,
                sizeof(FScalar) * pMemBlocks->nSize );
        } else {
            FileAndMemOp( Op, f, pMemBlocks->pData, pMemBlocks->nSize,
                Rec, Temp );
            if ( Op == OP_Dot ){
                f += f2;
                f2 = f;
            }
        }
        Rec.Record.BaseOffset += sizeof(FScalar) * pMemBlocks->nSize;
        ++ pMemBlocks;
    };
};


void FDiisTargetMemoryBlockSet::SerializeTo( FRecord const &ResidualRec,
    FRecord const &AmplitudeRec, FStorageDevice const &Device, FMemoryStack &Memory )
{
    FScalar
        f = 1.0;
    FileAndMemOp( OP_WriteFile, f, &m_ResBlocks[0], m_ResBlocks.size(),
        FStorageBlock(Device,ResidualRec), Memory );
    FileAndMemOp( OP_WriteFile, f, &m_AmpBlocks[0], m_AmpBlocks.size(),
        FStorageBlock(Device,AmplitudeRec), Memory );
}

// another evil hack...  FileAndMemOp always takes non-const, since some
// operations actually change the record contents.
static inline FStorageBlock StorageBlockFromConst( FStorageDevice const &Device,
    FStorageDevice::FRecord const &Record )
{
    return FStorageBlock( const_cast<FStorageDevice&>( Device ),
        const_cast<FStorageDevice::FRecord&>( Record ) );
}

void FDiisTargetMemoryBlockSet::CalcResidualDots( FScalar Out[],
    FRecord const Res[], uint nVectors, uint iSkip,
    FStorageDevice const &Device, FMemoryStack &Memory )
{
    for ( uint iVec = 0; iVec < nVectors; ++ iVec )
    {
        if ( iVec == iSkip )
            continue;
        FileAndMemOp( OP_Dot, Out[iVec], &m_ResBlocks[0], m_ResBlocks.size(),
            StorageBlockFromConst(Device,Res[iVec]), Memory );
    }
};

void FDiisTargetMemoryBlockSet::InterpolateBlockSet( FMemoryBlockSet &BlockSet,
        FScalar fOwnCoeff, FScalar const Coeffs[], FRecord const Vecs[],
        uint nVectors, uint iSkip, FStorageDevice const &Device, FMemoryStack &Memory )
{
    // scale this vector with given coefficient
    FMemoryBlockSet::const_iterator
        itBlock;
    _for_each( itBlock, BlockSet )
        Scale( itBlock->pData, fOwnCoeff, itBlock->nSize );

    // add scaled other vectors to current one.
    for ( uint iVec = 0; iVec < nVectors; ++ iVec )
    {
        if ( iVec == iSkip )
            continue;
        FScalar
            Ci = Coeffs[iVec];
        if ( Ci == 0.0 )
            continue;
        FileAndMemOp( OP_AddToMem, Ci, &BlockSet[0], BlockSet.size(),
            StorageBlockFromConst(Device,Vecs[iVec]), Memory );
    }
};

void FDiisTargetMemoryBlockSet::InterpolateFrom( FScalar fOwnCoeff, FScalar const Coeffs[],
    FRecord const Res[], FRecord const Amps[], uint nVectors, uint iSkip,
    FStorageDevice const &Device, FMemoryStack &Memory )
{
    InterpolateBlockSet( m_AmpBlocks, fOwnCoeff, Coeffs, Amps, nVectors, iSkip, Device, Memory );
    InterpolateBlockSet( m_ResBlocks, fOwnCoeff, Coeffs, Res, nVectors, iSkip, Device, Memory );
};



} // namespace ct


// kate: space-indent on; tab-indent on; backspace-indent on; tab-width 4; indent-width 4; mixedindent off; indent-mode normal;
