/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#include "BlockContract.h"


typedef FArrayOffset
    FStride;

template<bool AddToDest, bool UseFactor>
inline void ReorderBaseOpT( FScalar &Out, FScalar &In, FScalar const &DataFactor ) {
    if ( AddToDest && UseFactor )
        Out += In * DataFactor;
    else if ( !AddToDest && UseFactor )
        Out = In * DataFactor;
    else if ( AddToDest && !UseFactor )
        Out += In;
    else if ( !AddToDest && !UseFactor )
        Out = In;
}



// Out[i,j] <- In[i,j]
template<bool AddToDest, bool UseFactor>
void DirectMove2( FScalar *RESTRICT pOut_, FStride StrideOut,
    FScalar *RESTRICT pIn_, FStride StrideIn, FStride nRows, FStride nCols, FScalar const &DataFactor )
{
    FScalar
        *RESTRICT pIn = pIn_,
        *RESTRICT pOut = pOut_;
    for ( FStride nCol = 0; nCol < nCols; ++ nCol ){
        for ( FStride i = 0; i < (nRows >> 2); ++ i ){
            ReorderBaseOpT<AddToDest,UseFactor>(pOut[0], pIn[0], DataFactor );
            ReorderBaseOpT<AddToDest,UseFactor>(pOut[1], pIn[1], DataFactor );
            ReorderBaseOpT<AddToDest,UseFactor>(pOut[2], pIn[2], DataFactor );
            ReorderBaseOpT<AddToDest,UseFactor>(pOut[3], pIn[3], DataFactor );
            pOut += 4; pIn += 4;
        }
        for ( FStride i = 0; i < (nRows & 3); ++ i ){
            ReorderBaseOpT<AddToDest,UseFactor>( pOut[0], pIn[0], DataFactor );
            pOut += 1; pIn += 1;
        }
        pIn += StrideIn - nRows;
        pOut += StrideOut - nRows;
    }
}

// // Out[i,j] <- In[j,i]
// template<bool AddToDest, bool UseFactor>
// void TransposeMove2( FScalar *RESTRICT pOut_, FStride StrideOut,
//     FScalar *RESTRICT pIn_, FStride StrideIn, FStride nRows, FStride nCols, FScalar const &DataFactor )
// {
//     FScalar
//         *RESTRICT pIn = pIn_,
//         *RESTRICT pOut = pOut_;
//     FStride
//         a = StrideIn, b = a + StrideIn, c = b + StrideIn, d = c + StrideIn;
//     for ( FStride nCol = 0; nCol < nCols; ++ nCol ){
//         FStride nRow = 0;
//         for ( nRow = 0; nRow + 3 < nRows; nRow += 4 ){
//             ReorderBaseOpT<AddToDest,UseFactor>(pOut[0], pIn[0], DataFactor);
//             ReorderBaseOpT<AddToDest,UseFactor>(pOut[1], pIn[a], DataFactor);
//             ReorderBaseOpT<AddToDest,UseFactor>(pOut[2], pIn[b], DataFactor);
//             ReorderBaseOpT<AddToDest,UseFactor>(pOut[3], pIn[c], DataFactor);
//             pOut += 4; pIn += d;
//         }
//         for ( ; nRow < nRows; ++ nRow ){
//             ReorderBaseOpT<AddToDest,UseFactor>(pOut[0], pIn[0], DataFactor);
//             pOut += 1; pIn += a;
//         }
//         pIn += 1 - nRows * a;
//         pOut += StrideOut - nRows;
//     }
// }


// // Out[i,j] <- In[j,i]
// template<bool AddToDest, bool UseFactor>
// void TransposeMove2( FScalar *RESTRICT pOut_, FStride StrideOut,
//     FScalar *RESTRICT pIn_, FStride StrideIn, FStride nRows, FStride nCols, FScalar const &DataFactor )
// {
//     FStride
//         a = StrideIn, b = a + a, c = b + a, d = c + a;
//     FStride const
//         nBlk1 = 50;
//     for ( FStride nRowStart = 0; nRowStart < nRows; nRowStart += nBlk1 ) {
//         FStride
//             nRowLen = std::min(nRows - nRowStart, nBlk1);
//         FScalar
//             *RESTRICT pIn = pIn_ + nRowStart * a,
//             *RESTRICT pOut = pOut_ + nRowStart;
//
//         for ( FStride nCol = 0; nCol < nCols; ++ nCol ){
//             FStride nRow = 0;
//             for ( nRow = 0; nRow + 3 < nRowLen; nRow += 4 ){
//                 ReorderBaseOpT<AddToDest,UseFactor>(pOut[0], pIn[0], DataFactor);
//                 ReorderBaseOpT<AddToDest,UseFactor>(pOut[1], pIn[a], DataFactor);
//                 ReorderBaseOpT<AddToDest,UseFactor>(pOut[2], pIn[b], DataFactor);
//                 ReorderBaseOpT<AddToDest,UseFactor>(pOut[3], pIn[c], DataFactor);
//                 pOut += 4; pIn += d;
//             }
//             for ( ; nRow < nRowLen; ++ nRow ){
//                 ReorderBaseOpT<AddToDest,UseFactor>(pOut[0], pIn[0], DataFactor);
//                 pOut += 1; pIn += a;
//             }
//             pIn += 1 - nRowLen * a;
//             pOut += StrideOut - nRowLen;
//         }
//     }
// }



// // Out[i,j] <- In[j,i]
// template<bool AddToDest, bool UseFactor>
// void TransposeMove2( FScalar *RESTRICT pOut_, FStride StrideOut,
//     FScalar *RESTRICT pIn_, FStride StrideIn, FStride nRows, FStride nCols, FScalar const &DataFactor )
// {
//     FStride
//         a = StrideIn, b = a + a, c = b + a, d = c + a;
//     FStride const
//         nBlk1 = 50,
//         nBlk2 = nBlk1 * 20;
//     for ( FStride nColStart = 0; nColStart < nCols; nColStart += nBlk2 ) {
//         FStride
//             nColLen = std::min(nCols - nColStart, nBlk2);
//         for ( FStride nRowStart = 0; nRowStart < nRows; nRowStart += nBlk1 ) {
//             FStride
//                 nRowLen = std::min(nRows - nRowStart, nBlk1);
//             FScalar
//                 *RESTRICT pIn = pIn_ + nRowStart * a + nColStart,
//                 *RESTRICT pOut = pOut_ + nRowStart + nColStart * StrideOut;
//
//             for ( FStride nCol = 0; nCol < nColLen; ++ nCol ){
//                 FStride nRow = 0;
//                 for ( nRow = 0; nRow + 3 < nRowLen; nRow += 4 ){
//                     ReorderBaseOpT<AddToDest,UseFactor>(pOut[0], pIn[0], DataFactor);
//                     ReorderBaseOpT<AddToDest,UseFactor>(pOut[1], pIn[a], DataFactor);
//                     ReorderBaseOpT<AddToDest,UseFactor>(pOut[2], pIn[b], DataFactor);
//                     ReorderBaseOpT<AddToDest,UseFactor>(pOut[3], pIn[c], DataFactor);
//                     pOut += 4; pIn += d;
//                 }
//                 for ( ; nRow < nRowLen; ++ nRow ){
//                     ReorderBaseOpT<AddToDest,UseFactor>(pOut[0], pIn[0], DataFactor);
//                     pOut += 1; pIn += a;
//                 }
//                 pIn += 1 - nRowLen * a;
//                 pOut += StrideOut - nRowLen;
//             }
//         }
//     }
// }



template<bool AddToDest, bool UseFactor>
void TransposeMove2x2( FScalar *RESTRICT pOut_, FStride const *pStrideOut,
    FScalar *RESTRICT pIn_, FStride const *pStrideIn, FStride nRows, FStride nCols, FScalar const &DataFactor )
{
    FStride
        a = pStrideIn[0], b = a + a, c = b + a, d = c + a,
        u = pStrideOut[0], v = u + u, w = v + u, x = w + u;
    FStride const
        // blocking over both dimensions because we don't know which
        // is the fast one, if any should be.
        nBlk1 = 50,
        nBlk2 = nBlk1;//20 * nBlk1;
    for ( FStride nColStart = 0; nColStart < nCols; nColStart += nBlk2 ) {
        FStride
            nColLen = std::min(nCols - nColStart, nBlk2);
        for ( FStride nRowStart = 0; nRowStart < nRows; nRowStart += nBlk1 ) {
            FStride
                nRowLen = std::min(nRows - nRowStart, nBlk1),
                nIncIn = pStrideIn[1] - nRowLen * a,
                nIncOut = pStrideOut[1] - nRowLen * u;
            FScalar
                *RESTRICT pIn = pIn_ + nRowStart * a + nColStart * pStrideIn[1],
                *RESTRICT pOut = pOut_ + nRowStart * u + nColStart * pStrideOut[1];

            for ( FStride nCol = 0; nCol < nColLen; ++ nCol ){
                FStride nRow = 0;
                for ( nRow = 0; nRow + 3 < nRowLen; nRow += 4 ){
                    ReorderBaseOpT<AddToDest,UseFactor>(pOut[0], pIn[0], DataFactor);
                    ReorderBaseOpT<AddToDest,UseFactor>(pOut[u], pIn[a], DataFactor);
                    ReorderBaseOpT<AddToDest,UseFactor>(pOut[v], pIn[b], DataFactor);
                    ReorderBaseOpT<AddToDest,UseFactor>(pOut[w], pIn[c], DataFactor);
                    pOut += x; pIn += d;
                }
                for ( ; nRow < nRowLen; ++ nRow ){
                    ReorderBaseOpT<AddToDest,UseFactor>(pOut[0], pIn[0], DataFactor);
                    pOut += u; pIn += a;
                }
                pIn += nIncIn;
                pOut += nIncOut;
            }
        }
    }
}





template<bool AddToDest, bool UseFactor>
static void ReorderData1B( FScalar *RESTRICT pOut, FScalar *RESTRICT pIn,
    FStride const *pSize, FStride const *pStrideOut, FStride const *pStrideIn,
    FScalar const &DataFactor, uint n )
{
    if ( n == 1 ) {
        assert( n == 1 );
        FStride
            nRow = 0,
            StrideIn = *pStrideIn,
            StrideOut = *pStrideOut;
/*        if ( StrideIn * StrideOut == 1 ) {
            for ( nRow = 0; nRow + 3 < *pSize; nRow += 4 ){
                ReorderBaseOpT<AddToDest,UseFactor>(pOut[0], pIn[0], DataFactor);
                ReorderBaseOpT<AddToDest,UseFactor>(pOut[1], pIn[1], DataFactor);
                ReorderBaseOpT<AddToDest,UseFactor>(pOut[2], pIn[2], DataFactor);
                ReorderBaseOpT<AddToDest,UseFactor>(pOut[3], pIn[3], DataFactor);
                pOut += 4; pIn += 4;
            }
            for ( FStride i = 0; i < *pSize - nRow; ++ i )
                ReorderBaseOpT<AddToDest,UseFactor>(pOut[i], pIn[i], DataFactor);
        } else {*/
            for ( nRow = 0; nRow + 3 < *pSize; nRow += 4 ){
                ReorderBaseOpT<AddToDest,UseFactor>(pOut[nRow*StrideOut], pIn[nRow * StrideIn], DataFactor);
                ReorderBaseOpT<AddToDest,UseFactor>(pOut[(nRow+1) * StrideOut], pIn[(nRow+1) * StrideIn], DataFactor);
                ReorderBaseOpT<AddToDest,UseFactor>(pOut[(nRow+2) * StrideOut], pIn[(nRow+2) * StrideIn], DataFactor);
                ReorderBaseOpT<AddToDest,UseFactor>(pOut[(nRow+3) * StrideOut], pIn[(nRow+3) * StrideIn], DataFactor);
            }
            for ( ; nRow < *pSize; ++ nRow )
                ReorderBaseOpT<AddToDest,UseFactor>(pOut[nRow * StrideOut], pIn[nRow * StrideIn], DataFactor);
//         }
    } else if ( n == 2 ){
        if ( *(pStrideOut-1) == 1 && *(pStrideIn-1) == 1 )
            return DirectMove2<AddToDest,UseFactor>( pOut, *(pStrideOut),
                pIn, *(pStrideIn), *(pSize-1), *(pSize), DataFactor );
        else
            return TransposeMove2x2<AddToDest,UseFactor>( pOut, pStrideOut-1,
                pIn, pStrideIn-1, *(pSize-1), *(pSize), DataFactor );
// TODO: ^- try these again with better compilers or if treating large problem sizes.
//          for matrices in the 100..200 dimension the cache blocking simply does not work,
//          and we're better of with the smaller code size.
//     } else if ( n == 2 && *(pStrideOut-1) == 1 && *(pStrideIn-1) == 1 ){
//         return DirectMove2<AddToDest,UseFactor>( pOut, *(pStrideOut),
//             pIn, *(pStrideIn), *(pSize-1), *(pSize), DataFactor );
    } else {
        for ( FStride i = 0; i < *pSize; ++ i )
            ReorderData1B<AddToDest,UseFactor>(
                pOut + i * (*pStrideOut),
                pIn + i * (*pStrideIn),
                pSize - 1, pStrideOut - 1, pStrideIn - 1, DataFactor, n - 1 );
    }

}



typedef void (*FReorderDataFn)( FScalar *RESTRICT pOut, FScalar *RESTRICT pIn,
        FStride const *pSize, FStride const *pStrideOut, FStride const *pStrideIn,
        FScalar const &DataFactor, uint n );

#define FN(AddToDest,UseFactor) ReorderData1B<AddToDest,UseFactor>
static FReorderDataFn
    s_ReorderDataFns2[2][2] = { {FN(false,false),FN(false,true)}, {FN(true,false),FN(true,true)} };
//     s_ReorderDataFns2[2][2] = { {FN(false,true),FN(false,true)}, {FN(true,true),FN(true,true)} };
#undef REORDER_DATA_LINE
#undef FN



void ReorderData1A( FScalar *RESTRICT pOut, FScalar *RESTRICT pIn,
    FArraySizes const &StridesOut, FArraySizes const &Sizes, FScalar DataFactor,
    bool AddToDest, bool InToOut, FArraySizes const *pStridesIn )
{

    assert( StridesOut.size() >= Sizes.size() );

    // try to combine continuous dimenions of output array. This is possible
    // if iterating over indices n-dim Out(...,i,j,..) is equivalent to iterating
    // over (n-1)-dim Out(...,i*j,...).
    FArraySizes
        StridesIn_,
        StridesOut_,
        Sizes_;
    FStride
        nStrideIn = 1;
    for ( uint i = 0; i < Sizes.size(); ++ i ){
        if ( Sizes[i] != 1 ){
            if ( !StridesOut_.empty() && StridesOut[i] == StridesOut_.back() * Sizes_.back() &&
                 (pStridesIn == 0 || (*pStridesIn)[i] == StridesIn_.back() * Sizes_.back()) ){
                // dimension can be combined with last dimension.
                Sizes_.back() *= Sizes[i];
            } else {
                Sizes_.push_back(Sizes[i]);
                StridesOut_.push_back(StridesOut[i]);
                if ( pStridesIn == 0 )
                    StridesIn_.push_back(nStrideIn);
                else
                    StridesIn_.push_back((*pStridesIn)[i]);
            }
            nStrideIn *= Sizes[i];
        }
/*        Sizes_.push_back(Sizes[i]);
        StridesOut_.push_back(StridesOut[i]);
        if ( pStridesIn == 0 )
            StridesIn_.push_back(nStrideIn);
        else
            StridesIn_.push_back((*pStridesIn)[i]);
        nStrideIn *= Sizes[i];*/
    };

    uint
        n = StridesOut_.size();

    if ( n == 0 ) {
        // moving a single scalar (0d-array)
        if (  AddToDest &&  InToOut ) { ReorderBaseOpT<true,true>( *pOut, *pIn, DataFactor ); return; }
        if ( !AddToDest &&  InToOut ) { ReorderBaseOpT<false,true>( *pOut, *pIn, DataFactor ); return; }
        if (  AddToDest && !InToOut ) { ReorderBaseOpT<true,true>( *pIn, *pOut, DataFactor ); return; }
        if ( !AddToDest && !InToOut ) { ReorderBaseOpT<false,true>( *pIn, *pOut, DataFactor ); return; }
        return;
    }

    // if input set has significant size, try to shuffle around the
    // strides and sizes to obtain more favorable memory access patterns.
    if ( n != 1 && nStrideIn >= 40 ) {

        // find fastest dimension in output array (optimally: 1, but might also
        // be some larger number). Note that due to the space combination we
        // should have at most one of these.
        uint iOutLin = 0;
        FStride iSmallest = StridesOut_[0];
        for ( uint i = 1; i < n; ++ i )
            if ( StridesOut_[i] < iSmallest ) {
                iSmallest = StridesOut_[i];
                iOutLin = i;
            }

        // unless output dimension is the first one (the most friendly case),
        // move it into the second position so that we have a series of interleaved
        // matrix transpositions in the two fastest loops.
        if ( iOutLin == 0 ) {
            // direct sequential move. Input data order probably more or less okay.
        } else {
            // Transpose1 move.
            if ( InToOut ) {
                // make fastest loop over pOut linear.
                std::swap(StridesIn_[iOutLin],StridesIn_[0]);
                std::swap(StridesOut_[iOutLin],StridesOut_[0]);
                std::swap(Sizes_[iOutLin],Sizes_[0]);
                // make linear pIn loop second-to-fastest.
                std::swap(StridesIn_[iOutLin],StridesIn_[1]);
                std::swap(StridesOut_[iOutLin],StridesOut_[1]);
                std::swap(Sizes_[iOutLin],Sizes_[1]);

            } else {
                // actual output parameter is pIn. Leave that one linear, but
                // move linear pOut strides to second-to-fastest dimension.
                std::swap(StridesIn_[iOutLin],StridesIn_[1]);
                std::swap(StridesOut_[iOutLin],StridesOut_[1]);
                std::swap(Sizes_[iOutLin],Sizes_[1]);
            }
        };

        // try to decrease memory jumping in the remaining dimensions somewhat.
        for ( uint i = 2 - ((iOutLin == 0)?1:0); i < Sizes_.size(); ++ i )
            for ( uint j = i + 1; j < Sizes_.size(); ++ j )
                if ( StridesIn_[j]+StridesOut_[j] < StridesIn_[i]+StridesOut_[i] ){
                    std::swap(StridesIn_[i],StridesIn_[j]);
                    std::swap(StridesOut_[i],StridesOut_[j]);
                    std::swap(Sizes_[i],Sizes_[j]);
                }
    }


    assert( n != 0 );

    if ( InToOut )
        s_ReorderDataFns2[AddToDest][DataFactor!=1.0]( pOut, pIn, &Sizes_[n-1], &StridesOut_[n-1], &StridesIn_[n-1], DataFactor, n );
    else
        s_ReorderDataFns2[AddToDest][DataFactor!=1.0]( pIn, pOut, &Sizes_[n-1], &StridesIn_[n-1], &StridesOut_[n-1], DataFactor, n );
}



// kate: space-indent on; tab-indent on; backspace-indent on; tab-width 4; indent-width 4; mixedindent off; indent-mode normal;
