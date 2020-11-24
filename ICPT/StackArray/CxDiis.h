/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#ifndef _CT8K_DIIS_H
#define _CT8K_DIIS_H

#include <cstddef>
#include "CxTypes.h"
#include "CxStorageDevice.h"
#include "CxFixedSizeArray.h"

namespace ct {

// General DIIS support. These ones are ported and extended from
// f12_shared.f90's PerformDiis function.
//
// Unfortunatelly, due to the requirement of supporting multiple
// memory blocks in DIIS, much of the simplicity of that implementation
// is lost :(.

static const uint
    // maximum allowed number of iterative subspace vectors
    nMaxDiisCapacity = 31;

struct
    FDiisTarget;


struct FDiisOptions
{
    uint
        // maximum DIIS dimension allowed
        nMaxDim;
    double
        // residual threshold for inclusion of a vector in the DIIS state.
        ResidualThresh;

    // note: can be implicitly constructed from an uint!
    FDiisOptions( uint nMaxDim_ = 6, double ResidualThresh_ = 1e6 )
        : nMaxDim( nMaxDim_ ), ResidualThresh( ResidualThresh_ )
    {};
};

// FDiisState contains the properties and state of an DIIS object.
//
// The standard use of this object looks like this:
//
//      double
//          *T, *R;
//      FStorageBlock
//          DiisBlock(StorageDevice, DiisRecord);
//      FDiisState
//          Diis( DiisBlock, MemoryStack, nMaxDiis );
//      for ( uint nIt = 0; nIt < nMaxIterations; ++nIt ){
//          // calculate R(T) somehow.
//          [...]
//
//          Diis( T, nSizeT, R, nSizeR );
//          if ( Diis.fLastResidual() < ThrVar ){
//              Converged = True;
//              break;
//          }
//      }
//
// More general usage scenarios can be realized by using the
// FDiisTarget class and explicitly giving storage records for
// the results.
class FDiisState
{
public:
    typedef double FScalar;

    // Memory: used for intermediates during the PerformDiis function.
    //    before and after, no memory is left on the stack.
    // Storage: If provided, subspace data is stored here. If not,
    //    subspace data is stored in memory.
    explicit FDiisState( FMemoryStack &Memory,
        FDiisOptions const &Options = FDiisOptions(),
        FStorageBlock const &Storage = FStorageBlock() );

    // discards previous iteration vectors, but does not clear records.
    void Reset();

    void PerformDiis( FDiisTarget &TR, FScalar W1 = 1.0 );
    void operator () ( FDiisTarget &TR ){ PerformDiis(TR); }

    // convenience function to run DIIS on a block set in memory.
    // pOth: set of amplitudes to extrapolate, but not to include in the
    // DIIS system itself.
    void operator () ( double *pAmp, size_t nAmpSize, double *pRes,
        size_t nResSize = static_cast<size_t>(-1), double *pOth = 0, size_t nOthSize = 0 );

    double fLastResidual() const { return m_LastResidualNormSq; }
    double fLastCoeff() const { return m_LastAmplitudeCoeff; }
    uint nLastDim() const;
    uint nNextVec() const { return m_iNext; }
    uint nMaxDim() const { return m_Options.nMaxDim; };
private:
    FDiisOptions
        m_Options;
    FStorageBlock
        // where we store our intermediate data
        m_Storage;
    FMemoryStack
        // from where we get temporary space
        &m_Memory;

    uint
        // 0xffff: no vector in this slot. Otherwise: number of iterations
        // the vector in this slot has already been inside the DIIS system.
        m_iVectorAge[nMaxDiisCapacity],
        m_iNext; // next vector to be overwritten. nDim+1 if nDim < nMaxDim.
    FStreamSize
        // lengths of the amplitude and residual vectors, respectively.
        m_nAmplitudeLength,
        m_nResidualLength;

    // find vectors which are not considered too bad for extrapolation purposes.
    void FindUsefulVectors(uint *iUsedVecs, uint &nDimUsed, double &fBaseScale, uint iThis);

    typedef TArrayFix<double, (nMaxDiisCapacity+1)*(nMaxDiisCapacity+1)>
        FDiisMatrixBase;
    struct FDiisMatrix : public FDiisMatrixBase
    {
        FDiisMatrix( uint nRows, uint nCols )
            : m_nRows(nRows), m_nCols(nCols)
        {
            resize( m_nRows * m_nCols );
            this->FDiisMatrixBase::operator = ( 0.0 );
        }
        inline uint nStride() { return m_nRows; };
        inline double& operator() ( uint nRow, uint nCol ){ return (*this)[m_nRows*nCol + nRow]; }
    private:
        uint
            m_nRows, m_nCols;
    };

    FDiisMatrix
        // storage for a nMaxDim x nMaxDim matrix. Always stride nStride, but
        // not always everything used. Keeps the resident parts of the DIIS
        // system: <res[i], res[j]>. Indexed by storage slot indices.
        m_ErrorMatrix;
    typedef TArrayFix<double, (nMaxDiisCapacity+1)>
        FDiisVector;
    FDiisVector
        m_Weights;

    static const FStreamSize
        iNotPresent = static_cast<FStreamSize>( -1 );

    FStorageDevice::FRecord AmplitudeRecord( uint iVec );
    FStorageDevice::FRecord ResidualRecord( uint iVec );
private:
    // the following variables are kept for informative/displaying purposes
    double
        // dot(R,R) of last residual vector fed into this state.
        m_LastResidualNormSq,
        // coefficient the actual new vector got in the last DIIS step
        m_LastAmplitudeCoeff;

    // this one is used if the DIIS state is constructed in thin
    // air (so to say).
    FStorageDeviceMemoryBuf
        m_StorageDeviceToUseIfNoneProvided;
private:
    FDiisState( FDiisState const & ); // not implemented
    void operator = ( FDiisState const & ); // not implemented
};


// Represents a pair of (vector,residual).
//
// Note that there are simple-to use default implementations for
// one-memory-block, multiple-memory-blocks and lists of associated FTensor
// objects!
//
// For all functions in here, the 'Memory' parameter just supplies an object
// from which temporary memory may be drawn if required for the given
// operation.
struct FDiisTarget
{
    typedef double
        FScalar;
    typedef FStorageDevice::FRecord
        FRecord;

    // called at begin/end of diis routine. If required, may be used to
    // load/store information. Default implementation does nothing.
    virtual void Prepare();
    virtual void Finish();

    // return length of serialized residual in bytes.
    virtual FStreamSize nResidualLength() = 0;
    virtual FStreamSize nAmplitudeLength() = 0;

    // return dot(ThisR,ThisR) for *this residual.
    virtual double OwnResidualDot() = 0;

    // store residual and amplitude vectors to Device at
    // positions ResidualRec/AmplitudeRec.
    virtual void SerializeTo( FRecord const &ResidualRec,
        FRecord const &AmplitudeRec, FStorageDevice const &Device,
        FMemoryStack &Memory ) = 0;

    // build Out[i] = (R[i],ThisR) for other residuals stored at Device/Res[i],
    // except for i == iSkip.
    virtual void CalcResidualDots( FScalar Out[],
        FRecord const Res[], uint nVectors, uint iSkip,
        FStorageDevice const &Device, FMemoryStack &Memory ) = 0;

    // build ThisX = fOwnCoeff * ThisX + \sum_i Coeffs[i] X[i]
    // where X denotes amplitudes and residuals,
    // except for i == iSkip.
    virtual void InterpolateFrom( FScalar fOwnCoeff, FScalar const Coeffs[],
        FRecord const Res[], FRecord const Amps[], uint nVectors, uint iSkip,
        FStorageDevice const &Device, FMemoryStack &Memory ) = 0;

    virtual ~FDiisTarget();
};










struct FMemoryBlock{
    typedef FDiisTarget::FScalar
        FScalar;
    FScalar
        *pData;
    size_t
        nSize;
    FMemoryBlock(){};
    FMemoryBlock( FScalar *p_, size_t n_ ) : pData(p_), nSize(n_) {};
};



struct FDiisTargetMemoryBlockSet : public FDiisTarget
{
public:
    typedef std::size_t
        size_t;

    // construct an empty target: memory regions need to be added by
    // AddBlock()
    FDiisTargetMemoryBlockSet() {};
    // construct a one-block target.
    FDiisTargetMemoryBlockSet( FScalar *pAmp, size_t nAmpSize,
            FScalar *pRes, size_t nResSize )
    {
        AddBlockPair( pAmp, nAmpSize, pRes, nResSize );
    };
    ~FDiisTargetMemoryBlockSet(); // override

    // add pAmp/pRes to controlled block set.
    void AddBlockPair( FScalar *pAmp, size_t nAmpSize,
            FScalar *pRes, size_t nResSize )
    {
        AddAmpBlock( pAmp, nAmpSize );
        AddResBlock( pRes, nResSize );
    };
    void AddResBlock( FScalar *pBegin, size_t nSize ){
        m_ResBlocks.push_back( FMemoryBlock(pBegin, nSize) );
    }
    void AddAmpBlock( FScalar *pBegin, size_t nSize ){
        m_AmpBlocks.push_back( FMemoryBlock(pBegin, nSize) );
    }

public:
    FStreamSize nResidualLength(); // override
    FStreamSize nAmplitudeLength(); // override

    double OwnResidualDot(); // override

    void SerializeTo( FRecord const &ResidualRec,
        FRecord const &AmplitudeRec, FStorageDevice const &Device,
        FMemoryStack &Memory ); // override

    void CalcResidualDots( FScalar Out[],
        FRecord const Res[], uint nVectors, uint iSkip,
        FStorageDevice const &Device, FMemoryStack &Memory ); // override

    void InterpolateFrom( FScalar fOwnCoeff, FScalar const Coeffs[],
        FRecord const Res[], FRecord const Amps[], uint nVectors, uint iSkip,
        FStorageDevice const &Device, FMemoryStack &Memory ); // override

protected:
    static const uint
        nMaxMemoryBlocks = 16;
        //nMaxMemoryBlocks = 128;
    typedef TArrayFix<FMemoryBlock, nMaxMemoryBlocks>
        FMemoryBlockSet;
    FMemoryBlockSet
        m_AmpBlocks,
        m_ResBlocks;

    static void InterpolateBlockSet( FMemoryBlockSet &BlockSet,
        FScalar fOwnCoeff, FScalar const Coeffs[], FRecord const Vecs[],
        uint nVectors, uint iSkip, FStorageDevice const &Device, FMemoryStack &Memory );
    static FStreamSize nBlockSetLength( FMemoryBlockSet const &BlockSet );
};

typedef FDiisTargetMemoryBlockSet
    FDiisTarget1; // most common one, likely...


} // namespace ct


#endif // _CT8K_DIIS_H

// kate: space-indent on; tab-indent on; backspace-indent on; tab-width 4; indent-width 4; mixedindent off; indent-mode normal;
