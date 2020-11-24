/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#include <stdexcept>
#include <stdio.h> // for tmpfile and c-style file handling functions
#include <memory.h>
#include <string.h>

#include "CxStorageDevice.h"
#include "CxMemoryStack.h"
#include "CxAlgebra.h"

namespace ct {

FStorageBlock FStorageDevice::AllocNewBlock(uint64_t SizeInBytes)
{
    FRecord r = AllocNewRecord(SizeInBytes);
    return FStorageBlock(*this, r);
};

FStorageDevice::~FStorageDevice()
{
};



FRecord FStorageDevicePosixFs::AllocNewRecord(uint64_t SizeInBytes)
{
    uint
        NewId = FileIds.size();
    FILE *NewFile = tmpfile();
    if ( NewFile == 0 )
        throw std::runtime_error("FStorageDevicePosixFs: Failed to open a temporary file via tmpfile() function.");
    FileIds[NewId] = NewFile;
    return FRecord(NewId, 0, 0, SizeInBytes);
};

void FStorageDevicePosixFs::Write( FRecord const &r, void const *pData, FStreamOffset nLength, FStreamOffset Offset ) const
{
    FILE *File = GetHandle(r);
    fseek(File, Offset + r.BaseOffset, SEEK_SET);
    fwrite(pData, 1, nLength, File);
    if (ferror(File)) throw std::runtime_error("failed to write data.");
};

void FStorageDevicePosixFs::Read( FRecord const &r, void *pData, FStreamOffset nLength, FStreamOffset Offset ) const
{
    FILE *File = GetHandle(r);
    fseek(File, Offset + r.BaseOffset, SEEK_SET);
    fread(pData, 1, nLength, File);
    if (ferror(File)) throw std::runtime_error("failed to read data.");
};

void FStorageDevicePosixFs::Reserve( FRecord const &r, FStreamOffset nLength ) const
{
};

void FStorageDevicePosixFs::Delete( FRecord const &r )
{
    FILE *File = GetHandle(r);
    fclose(File);
    FileIds.erase(r.iFile);
};

FILE *FStorageDevicePosixFs::GetHandle(FRecord const &r) const {
    FFileIdMap::const_iterator
        it = FileIds.find(r.iFile);
    if ( it == FileIds.end() )
        throw std::runtime_error("FStorageDevicePosixFs: Attempted to access a non-exitent file id.");
    return it->second;
};


// FILE *FStorageDevicePosixFs::GetHandle(FRecord const &r) {
//     return const_cast<FILE*>( const_cast<FStorageDevicePosixFs const*>(this)->GetHandle(r) );
// };


FStorageDevicePosixFs::FStorageDevicePosixFs()
{
};


FStorageDevicePosixFs::~FStorageDevicePosixFs()
{
    FFileIdMap::reverse_iterator
        itFile;
    for (itFile = FileIds.rbegin(); itFile != FileIds.rend(); ++ itFile)
        fclose(itFile->second);
};


void FStorageDevice::FRecord::SetLength(FStreamOffset NewLength) const
{
    assert(NewLength != UnknownSize);
    assert(EndOffset == UnknownSize || EndOffset - BaseOffset == NewLength);
    const_cast<FStorageDevice::FRecord*>(this)->EndOffset = BaseOffset + NewLength;
}








FStorageDeviceMemoryBuf::FBuffer &FStorageDeviceMemoryBuf::GetBuf(FRecord const &r) const {
    assert(r.iRecord < m_Buffers.size());
    return *(const_cast<FBuffer*>(&m_Buffers[r.iRecord]));
}

FRecord FStorageDeviceMemoryBuf::AllocNewRecord(uint64_t SizeInBytes)
{
    m_Buffers.push_back(FBuffer());
    m_Buffers.back().p = 0;
    m_Buffers.back().Length = 0;
    FRecord
        r = FRecord(0, m_Buffers.size() - 1, 0);
    if ( SizeInBytes != 0 )
        Reserve(r, SizeInBytes);
    return r;
};

void FStorageDeviceMemoryBuf::Write( FRecord const &r, void const *pData, FStreamOffset nLength, FStreamOffset Offset ) const
{
    FBuffer &Buf = GetBuf(r);
    if ( Buf.p == 0 )
        Reserve(r, Offset + nLength);

    Offset += r.BaseOffset;
    assert(Offset + nLength <= Buf.Length );
    memcpy(Buf.p + Offset, pData, nLength);
};

void FStorageDeviceMemoryBuf::Read( FRecord const &r, void *pData, FStreamOffset nLength, FStreamOffset Offset ) const
{
    FBuffer &Buf = GetBuf(r);
    Offset += r.BaseOffset;
    assert(Offset + nLength <= Buf.Length );
    memcpy(pData, Buf.p + Offset, nLength);
};

void FStorageDeviceMemoryBuf::Reserve( FRecord const &r, FStreamOffset nLength ) const
{
    FBuffer &Buf = GetBuf(r);
    delete Buf.p;
    Buf.p = static_cast<char*>(malloc(nLength));
    Buf.Length = nLength;
};

void FStorageDeviceMemoryBuf::Delete( FRecord const &r )
{
    FBuffer &Buf = GetBuf(r);
    delete Buf.p;
    Buf.p = 0;
    Buf.Length = 0;
};


FStorageDeviceMemoryBuf::~FStorageDeviceMemoryBuf()
{
    // semi-safe...
    for ( uint i = m_Buffers.size(); i != 0; -- i )
        delete m_Buffers[i - 1].p;
    m_Buffers.clear();
};








template<class FScalar>
void FileAndMemOp( FFileAndMemOp Op, FScalar &F, FScalar *pMemBuf,
        std::size_t nMemLength, FStorageBlock const &Rec,
        FMemoryStack &Temp )
{
    // perform simple memory block/file block operations with
    // constant memory buffer size.
    std::size_t
        nSize,
        nOff = 0,
        nFileStep = std::min( Temp.MemoryLeft(), (size_t)1000000 ); // ! ~ 1 mb
    FScalar
        fVal = 0.0,
        *pFileBuf;
    const uint
        Byte = sizeof(FScalar);
    nFileStep = nFileStep/Byte - 1;
    //assert_rt( nFileStep != 0 );
    Temp.Alloc( pFileBuf, nFileStep );

    while ( nOff < nMemLength ) {
        nSize = std::min( nFileStep, nMemLength - nOff );
        Rec.pDevice->Read( Rec.Record, pFileBuf, nSize*Byte, nOff*Byte );

        switch( Op ){
            case OP_AddToMem:
                // add f*file_content to BufM.
                Add( pMemBuf + nOff, pFileBuf, F, nSize );
                break;
            case OP_AddToFile:
                // add f*mem_content to file
                Add( pFileBuf, pMemBuf + nOff, F, nSize );
                Rec.pDevice->Write( Rec.Record, pFileBuf, nSize*Byte, nOff*Byte );
                break;
            case OP_Dot:
                // form dot(mem,file), store result in F.
                fVal += Dot( pFileBuf, pMemBuf + nOff, nSize );
                F = fVal;
                break;
            default:
                assert(0);
        }
        nOff += nSize;
    }

    Temp.Free( pFileBuf );
}


template void FileAndMemOp( FFileAndMemOp, double&, double*, std::size_t, FStorageBlock const &, FMemoryStack & );
// template void FileAndMemOp( FFileAndMemOp, float&, float*, std::size_t, FStorageBlock const &, FMemoryStack & );

} // namespace ct
