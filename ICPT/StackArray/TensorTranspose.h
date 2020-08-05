/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */



void ReorderData1A( FScalar *RESTRICT pOut, FScalar *RESTRICT pIn,
    FArraySizes const &StridesOut, FArraySizes const &Sizes, FScalar DataFactor,
    bool AddToDest, bool InToOut, FArraySizes const *pStridesIn );

