/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#ifndef CX_NUMPY_ARRAY_H
#define CX_NUMPY_ARRAY_H

#include <stdio.h> // for FILE*.
#include <stdexcept>
#include "CxPodArray.h"
#include "CxDefs.h"
// support for reading and writing array data in .npy format.
// (that's what numpy.save() and numpy.load() use)

class FNdArrayView;

namespace ct {


struct FIoExceptionNpy : public std::runtime_error {
   explicit FIoExceptionNpy(std::string const &Msg, std::string const &FileName = "");
};

typedef TArray<std::size_t>
   FShapeNpy;

// write the continuous array pData[i,j,k,..] to the given file in .npy
// format. pShape[i] gives the number of indices in dimension #i, nDim
// gives the total dimension of the array.
void WriteNpy(FILE *File, FScalar const *pData, std::size_t const pShape[], std::size_t nDim);
void WriteNpy(std::string const &FileName, FScalar const *pData, std::size_t const pShape[], std::size_t nDim);
void WriteNpy(FILE *File, FScalar const *pData, FShapeNpy const &Shape);
void WriteNpy(std::string const &FileName, FScalar const *pData, FShapeNpy const &Shape);

// convenience functions for making array shape objects.
FShapeNpy MakeShape(std::size_t i0);
FShapeNpy MakeShape(std::size_t i0, std::size_t i1);
FShapeNpy MakeShape(std::size_t i0, std::size_t i1, std::size_t i2);
FShapeNpy MakeShape(std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3);

struct FArrayNpy {
   TArray<FScalar>
      Data;
   FShapeNpy
      Shape,
      Strides; // <- may be inverted when reading in stuff in C order.
   std::size_t Rank() const { return Shape.size(); }
   std::size_t Size() const;

   enum FCreationFlags {
      NPY_OrderFortran = 0x0, // first dimension first
      NPY_OrderC = 0x1, // last dimension first
      NPY_ClearData = 0x2 // set data to zero
   };

   FArrayNpy() {}
   explicit FArrayNpy(FShapeNpy const &Shape, uint Flags = NPY_OrderC);

   void swap(FArrayNpy &Other);

   FScalar &operator() (std::size_t i0) { return Data[Strides[0]*i0]; }
   FScalar &operator() (std::size_t i0, std::size_t i1) { return Data[Strides[0]*i0 + Strides[1]*i1]; }
   FScalar &operator() (std::size_t i0, std::size_t i1, std::size_t i2) { return Data[Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2]; }
   FScalar &operator() (std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3) { return Data[Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2 + Strides[3]*i3]; }

   FScalar operator() (std::size_t i0) const { return Data[Strides[0]*i0]; }
   FScalar operator() (std::size_t i0, std::size_t i1) const  { return Data[Strides[0]*i0 + Strides[1]*i1]; }
   FScalar operator() (std::size_t i0, std::size_t i1, std::size_t i2) const { return Data[Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2]; }
   FScalar operator() (std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3) const  { return Data[Strides[0]*i0 + Strides[1]*i1 + Strides[2]*i2 + Strides[3]*i3]; }

   // flags: bit field of NPY_*.
   void Init(FShapeNpy const &Shape_, uint Flags = NPY_OrderC);
};

// read array from File in .npy format.
void ReadNpy(FArrayNpy &Out, FILE *File);
void ReadNpyData(FNdArrayView &Out, FILE *File);
void ReadNpy(FArrayNpy &Out, std::string const &FileName);
void ReadNpyData(FNdArrayView &Out, std::string const &FileName);

} // namespace ct

#endif // CX_NUMPY_ARRAY_H
