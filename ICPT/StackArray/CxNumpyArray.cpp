/*                                                                                                                                                                         Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#include <stdint.h>
#include "CxNumpyArray.h"
#include "iostream"
#include "BlockContract.h"

#ifdef _MSC_VER
	#define snprintf _snprintf   // yes, I know that they are semnatically different, but it does not matter here.
								 // (snprintf is neither defined in C89 nor in C++98)
#endif

namespace ct {

void WriteNpy(FILE *File, FScalar const *pData, std::size_t const pShape[], std::size_t nDim)
{
   // magic string (6), version number (1.0), two bytes giving the header length
   // minus 10 Header length as pos [8],[9] will be written later.
#ifdef _SINGLE_PRECISION
   char const *BaseTemplate = "{'descr': '<f4', 'fortran_order': True, 'shape': (%s), }";
#else
   char const *BaseTemplate = "{'descr': '<f8', 'fortran_order': True, 'shape': (%s), }";
#endif
   char Header[256];
   char ShapeDesc[256];
   switch ( nDim ) {
      // why no loop? Because c string processing sucks and I don't want to deal with it.
#define I(n) (int)pShape[n]
      case 0: ShapeDesc[0] = 0; break;
      case 1: snprintf(&ShapeDesc[0], 256, "%i", I(0)); break;
      case 2: snprintf(&ShapeDesc[0], 256, "%i, %i", I(0), I(1)); break;
      case 3: snprintf(&ShapeDesc[0], 256, "%i,%i,%i", I(0), I(1), I(2)); break;
      case 4: snprintf(&ShapeDesc[0], 256, "%i,%i,%i,%i", I(0), I(1), I(2), I(3)); break;
      case 5: snprintf(&ShapeDesc[0], 256, "%i,%i,%i,%i,%i", I(0), I(1), I(2), I(3), I(4)); break;
      case 6: snprintf(&ShapeDesc[0], 256, "%i,%i,%i,%i,%i,%i", I(0), I(1), I(2), I(3), I(4), I(5)); break;
      default: throw FIoExceptionNpy("input array rank not supported.");
#undef I
   }
   for ( std::size_t i = 0; i < 10; ++ i )
      Header[i] = "\x93NUMPY\x01\x00~~"[i];
   int
      nw;
   nw = snprintf(&Header[10], 256-10, BaseTemplate, &ShapeDesc[0]);

   uint16_t
      HeaderLenTotal = 10 + nw,
      HeaderLen;
   // pad with spaces until data address starts at 16-byte boundary
   // (for alignment purposes: npy-format.txt forces that). And add a
   // newline.
   while((HeaderLenTotal + 1) % 16 != 0) {
      if (HeaderLenTotal >= 256-1) throw FIoExceptionNpy("programming error in header generation.");
      Header[HeaderLenTotal] = 0x20;
      HeaderLenTotal += 1;
   }
   HeaderLenTotal += 1;
   Header[HeaderLenTotal-1] = 0x0a;
   // calculate and store header lenght; -10: the magic string,
   // the version, and the header length itself are not included
   // in the count.
   HeaderLen = HeaderLenTotal - 10;
   Header[8] = HeaderLen & 0xff;
   Header[9] = HeaderLen >> 8;

   fwrite(&Header[0], 1, HeaderLenTotal, File);

   // now write actual data.
   std::size_t
      DataSize = 1, nWritten;
   for ( std::size_t iDim = 0; iDim < nDim; ++ iDim )
      DataSize *= pShape[iDim];
   nWritten = fwrite(pData, sizeof(pData[0]), DataSize, File);
   if ( nWritten != DataSize )
      throw FIoExceptionNpy("error in writing array data (disk full?)");
};


// read array from File in .npy format.
void ReadNpy(FArrayNpy &Out, FILE *File)
{
   char
      Header0[10],
      Header1[256];
   if ( 10 != fread(&Header0[0], 1, 10, File) )
      throw FIoExceptionNpy("Failed to read npy file initial header");
   for ( unsigned i = 0; i < 6; ++ i )
      if ( Header0[i] != "\x93NUMPY"[i] )
         throw FIoExceptionNpy("File does not have a numpy header.");
//    printf("Version: %i.%i\n", (int)Header0[6], (int)Header0[7]);
   if ( Header0[6] != 0x01 || Header0[7] != 0x00 )
      throw FIoExceptionNpy("Cannot read this file version (know only npy version 1.0).");
   uint16_t
      HeaderSize = (uint16_t)Header0[8] + (((uint16_t)Header0[9]) << 8);
   if ( HeaderSize >= 256 )
      throw FIoExceptionNpy("File header too large. Expected at most 256 bytes.");
   if ( HeaderSize != fread(&Header1[0], 1, HeaderSize, File) )
      throw FIoExceptionNpy("Failed to read npy file main header");

   // now comes the first complicated part: we need to read
   // the python dictionary containing the data declaration.
   // Looks like this:
   char const *BaseTemplate = "{'descr': '%63[^']', 'fortran_order': %15[^,], 'shape': (%255[^)]), }";
   // Note that the order of entries is actually fixed, as are the
   // positions of the spaces: npy-format.txt says this is generated
   // using the pformat function which enforces that.
   char
      // well... that's not safe.
      DataType[64],
      FortranOrder[16],
      ShapeDesc[256];
   int
      nItems;
   nItems = sscanf(&Header1[0], BaseTemplate, DataType, FortranOrder, ShapeDesc);
   if ( nItems != 3 )
      throw FIoExceptionNpy("Failed parse array shape description.");
#ifdef _SINGLE_PRECISION
   if ( strcmp(&DataType[0], "<f4") != 0 )
      throw FIoExceptionNpy("Can only read single precision arrays. Input has type: " + std::string(DataType) + "'");
#else
   if ( strcmp(&DataType[0], "<f8") != 0 )
      throw FIoExceptionNpy("Can only read double precision arrays. Input has type: " + std::string(DataType) + "'");
#endif
   bool
      OrderIsFortran = strcmp(&FortranOrder[0], "True") == 0,
      OrderIsC = strcmp(&FortranOrder[0], "False") == 0;
   if ( !OrderIsFortran && !OrderIsC )
      throw FIoExceptionNpy("fortan_order field misformed in input.");

   if (OrderIsC) {
     std::cout << "not fortran style while reading "<<std::endl;
     exit(0);
   }
   FShapeNpy
      Shape;
   Shape.clear();
   Shape.reserve(6);
   char
      *p = &ShapeDesc[0];
   for ( ; ; ) {
      if ( p[0] == 0 ) // should always be 0-terminated if we reached this place.
         break;
      int n;
      nItems = sscanf(p, "%i", &n);
      if ( nItems != 1 )
         throw FIoExceptionNpy("failed to process array shape description.");
      Shape.push_back(n);
      while ( p[0] != 0 && p[0] != ',' )
         p += 1;
      if ( p[0] == ',' )
         p += 1;
   }

   // now compute strides and read actual data.
   Out.Init(Shape, OrderIsFortran ? FArrayNpy::NPY_OrderFortran : FArrayNpy::NPY_OrderC);
   std::size_t
      nRead, DataSize = Out.Size();
   nRead = fread(&Out.Data[0], sizeof(Out.Data[0]), DataSize, File);
   if ( nRead != DataSize )
      throw FIoExceptionNpy("error in reading array data (file complete?)");
};


// read array from File in .npy format.
void ReadNpyData(FNdArrayView &Out, FILE *File)
{
   char
      Header0[10],
      Header1[256];
   if ( 10 != fread(&Header0[0], 1, 10, File) )
      throw FIoExceptionNpy("Failed to read npy file initial header");
   for ( unsigned i = 0; i < 6; ++ i )
      if ( Header0[i] != "\x93NUMPY"[i] )
         throw FIoExceptionNpy("File does not have a numpy header.");
//    printf("Version: %i.%i\n", (int)Header0[6], (int)Header0[7]);
   if ( Header0[6] != 0x01 || Header0[7] != 0x00 )
      throw FIoExceptionNpy("Cannot read this file version (know only npy version 1.0).");
   uint16_t
      HeaderSize = (uint16_t)Header0[8] + (((uint16_t)Header0[9]) << 8);
   if ( HeaderSize >= 256 )
      throw FIoExceptionNpy("File header too large. Expected at most 256 bytes.");
   if ( HeaderSize != fread(&Header1[0], 1, HeaderSize, File) )
      throw FIoExceptionNpy("Failed to read npy file main header");

   // now comes the first complicated part: we need to read
   // the python dictionary containing the data declaration.
   // Looks like this:
   char const *BaseTemplate = "{'descr': '%63[^']', 'fortran_order': %15[^,], 'shape': (%255[^)]), }";
   // Note that the order of entries is actually fixed, as are the
   // positions of the spaces: npy-format.txt says this is generated
   // using the pformat function which enforces that.
   char
      // well... that's not safe.
      DataType[64],
      FortranOrder[16],
      ShapeDesc[256];
   int
      nItems;
   nItems = sscanf(&Header1[0], BaseTemplate, DataType, FortranOrder, ShapeDesc);
   if ( nItems != 3 )
      throw FIoExceptionNpy("Failed parse array shape description.");
#ifdef _SINGLE_PRECISION
   if ( strcmp(&DataType[0], "<f4") != 0 )
      throw FIoExceptionNpy("Can only read single precision arrays. Input has type: " + std::string(DataType) + "'");
#else
   if ( strcmp(&DataType[0], "<f8") != 0 )
      throw FIoExceptionNpy("Can only read double precision arrays. Input has type: " + std::string(DataType) + "'");
#endif
   bool
      OrderIsFortran = strcmp(&FortranOrder[0], "True") == 0,
      OrderIsC = strcmp(&FortranOrder[0], "False") == 0;
   if ( !OrderIsFortran && !OrderIsC )
      throw FIoExceptionNpy("fortan_order field misformed in input.");

   Out.Sizes.clear();
   //Out.Sizes.reserve(6);
   char
      *p = &ShapeDesc[0];
   for ( ; ; ) {
      if ( p[0] == 0 ) // should always be 0-terminated if we reached this place.
         break;
      int n;
      nItems = sscanf(p, "%i", &n);
      if ( nItems != 1 )
         throw FIoExceptionNpy("failed to process array shape description.");
      Out.Sizes.push_back(n);
      while ( p[0] != 0 && p[0] != ',' )
         p += 1;
      if ( p[0] == ',' )
         p += 1;
   }

   Out.Strides.resize(Out.Sizes.size());

   std::size_t
     pDataSize = 1;
   if ( OrderIsFortran) {
     // fortran order.
     for ( std::size_t iDim = 0; iDim < Out.Sizes.size(); ++ iDim ) {
       Out.Strides[iDim] = pDataSize;
       pDataSize *= Out.Sizes[iDim];
     }
   } else {
     // that's the default case for npy files, unfortunately.
     for ( std::size_t iDim = 0; iDim < Out.Sizes.size(); ++ iDim ) {
       std::size_t iDim_ = Out.Sizes.size() - iDim - 1;
       Out.Strides[iDim_] = pDataSize;
       pDataSize *= Out.Sizes[iDim_];
     }
     //sometimes when the total dimension is only 1 python writes a c style 
     //tensor even if you explicitly ask it to write fortran style
     if (pDataSize != 1) {
       std::cout << "not fortran style while reading "<<std::endl;
       exit(0);
     }
   }
   
   if (pDataSize != Out.nValues()) 
     throw FIoExceptionNpy("numpy array not of correct size");
   
   std::size_t
     nRead;
   nRead = fread(&Out.pData[0], sizeof(Out.pData[0]), pDataSize, File);
   if ( nRead != pDataSize )
     throw FIoExceptionNpy("error in reading array data (file complete?)");
};


void WriteNpy(std::string const &FileName, FScalar const *pData, std::size_t const pShape[], std::size_t nDim)
{
   FILE *File = fopen(FileName.c_str(), "wb");
   if (File == 0)
      throw FIoExceptionNpy("Failed to open for writing", FileName);
   try {
      WriteNpy(File, pData, pShape, nDim);
   }
   catch ( FIoExceptionNpy &e ) {
     std::cout << "Trouble in reading file "<<FileName<<std::endl;
      throw FIoExceptionNpy(e.what(), FileName);
   }
   fclose(File);
}

void WriteNpy(FILE *File, FScalar const *pData, FShapeNpy const &Shape) {
   return WriteNpy(File, pData, &Shape[0], Shape.size());
}

void WriteNpy(std::string const &FileName, FScalar const *pData, FShapeNpy const &Shape) {
   return WriteNpy(FileName, pData, &Shape[0], Shape.size());
}

void ReadNpy(FArrayNpy &Out, std::string const &FileName)
{
   FILE *File = fopen(FileName.c_str(), "rb");
   if (File == 0)
      throw FIoExceptionNpy("Failed to open for reading", FileName);
   try {
      ReadNpy(Out, File);
   }
   catch ( FIoExceptionNpy &e ) {
      throw FIoExceptionNpy(e.what(), FileName);
   }
   fclose(File);
};

void ReadNpyData(FNdArrayView &Out, std::string const &FileName)
{
   FILE *File = fopen(FileName.c_str(), "rb");
   if (File == 0)
      throw FIoExceptionNpy("Failed to open for reading", FileName);
   try {
     ReadNpyData(Out, File);
   }
   catch ( FIoExceptionNpy &e ) {
      throw FIoExceptionNpy(e.what(), FileName);
   }
   fclose(File);
};



// convenience functions for making array shape objects.
FShapeNpy MakeShape(std::size_t i0)
   { FShapeNpy r(1); r[0] = i0; return r; }
FShapeNpy MakeShape(std::size_t i0, std::size_t i1)
   { FShapeNpy r(2); r[0] = i0; r[1] = i1; return r; }
FShapeNpy MakeShape(std::size_t i0, std::size_t i1, std::size_t i2)
   { FShapeNpy r(3); r[0] = i0; r[1] = i1; r[2] = i2; return r; }
FShapeNpy MakeShape(std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3)
   { FShapeNpy r(4); r[0] = i0; r[1] = i1; r[2] = i2; r[3] = i3; return r; }


FIoExceptionNpy::FIoExceptionNpy(std::string const &Msg, std::string const &FileName)
   : std::runtime_error(Msg + std::string((FileName != "")? (" while processing '" + FileName + "'").c_str():("")))
{};


std::size_t FArrayNpy::Size() const
{
   std::size_t Size = 1;
   for ( unsigned i = 0; i < Rank(); ++ i )
      Size *= Shape[i];
   return Size;
};

void FArrayNpy::swap(FArrayNpy &Other) {
   Data.swap(Other.Data);
   Shape.swap(Other.Shape);
   Strides.swap(Other.Strides);
}

void FArrayNpy::Init(FShapeNpy const &Shape_, uint Flags)
{
   Shape = Shape_;
   Strides.resize(Rank());

   std::size_t
      DataSize = 1;
   if ( (Flags & NPY_OrderC) == 0) {
      // fortran order.
      for ( std::size_t iDim = 0; iDim < Rank(); ++ iDim ) {
         Strides[iDim] = DataSize;
         DataSize *= Shape[iDim];
      }
   } else {
      // that's the default case for npy files, unfortunately.
      for ( std::size_t iDim = 0; iDim < Rank(); ++ iDim ) {
         std::size_t iDim_ = Rank() - iDim - 1;
         Strides[iDim_] = DataSize;
         DataSize *= Shape[iDim_];
      }
   }
   Data.resize(DataSize);
   if ((Flags & NPY_ClearData) != 0)
      Data.clear_data();
}

FArrayNpy::FArrayNpy(FShapeNpy const &Shape_, uint Flags)
{
   Init(Shape_, Flags);
}


} // namespace ct


