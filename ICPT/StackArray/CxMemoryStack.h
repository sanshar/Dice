/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#ifndef _CX_MEMORYSTACK_H
#define _CX_MEMORYSTACK_H

#include <cstddef> // for std::size_t
#include <string.h> // for memset

using std::size_t;

typedef unsigned int
   uint;

namespace ct {

/// An object managing a continuous block of memory. Used for allocation-free
/// temporary work space storage.
/// Notes:
///    - Deallocations must occur in reverse allocation order. A deallocation
///      FREES ALL MEMORY ALLOCATED AFTER THE ALLOCATION POINT.
///    - Both alloc and free are effectively non-ops in terms of computational
///      cost. Any work space, no matter how large or small, can be allocated
///      at will.
///    - There is no memory fragmentation.
struct FMemoryStack
{
    /// allocate nSizeInBytes bytes off the stack. Return pointer to start.
    virtual void* Alloc( size_t nSizeInBytes ) = 0;
    /// Free pointer p on stack. Should be an address returned by Alloc.
    virtual void Free( void *p ) = 0;
    /// return amount of memory left (in bytes).
    virtual size_t MemoryLeft() = 0;

    /// allocate nObjects objects of size sizeof(T). Store pointer in p.
    /// Usage:
    ///     MyType *p;
    ///     Mem.Alloc(p, 10);  // allocate array of 10 objects of type MyType.
    template<class T>
    inline void Alloc( T *&p, std::size_t nObjects = 1 ){
        p = reinterpret_cast<T*>( this->Alloc( sizeof(T) * nObjects ) );
        // note: objects are not constructed. Use only for PODs!
    }

    /// as Alloc(), but set allocated memory to zero.
    template<class T>
    inline void ClearAlloc( T *&p, std::size_t nObjects = 1 ){
        Alloc(p, nObjects);
        ::memset(p, 0, sizeof(T) * nObjects);
    }

    template<class T>
    inline T* AllocN(std::size_t nObjects, T const &){
        return reinterpret_cast<T*>( this->Alloc(sizeof(T) * nObjects) );
    }

    template<class T>
    inline T* ClearAllocN(std::size_t nObjects, T const &){
        T *p;
        this->Alloc(p, nObjects);
        ::memset(p, 0, sizeof(T) * nObjects);
        return p;
    }

    /// align stack to next 'Boundary' (must be power of two)-boundary .
    void Align( uint Boundary );

    virtual ~FMemoryStack();
};

/// A memory stack defined by a base pointer and a size. Can own
/// the memory managed, but can also be put on memory allocated from
/// elsewhere.
struct FMemoryStack2 : public FMemoryStack {
   /// creates a stack of 'nInitialSize' bytes on the global heap.
   /// If nInitialSize is 0, the stack is created empty and a storage
   /// area can be later assigned by AssignMemory() or Create().
   explicit FMemoryStack2( std::size_t nInitialSize = 0 ) : m_pData(0) { Create(nInitialSize); }
   /// if pBase != 0 && nActualSize >= nNeededSize:
   ///    creates a stack of 'nActualSize' bytes at memory pBase (i.e., at a memory location given from the outside)
   /// otherwise
   ///    construction equivalent to FMemoryStack2((nNeededSize != 0)? nNeededSize : nActualSize).
   FMemoryStack2( char *pBase, std::size_t nActualSize, std::size_t nNeededSize = 0 );
   inline void PushTopMark();
   inline void PopTopMark();

   void* Alloc( size_t nSizeInBytes ); // override
   using FMemoryStack::Alloc;
   void Free( void *p ); // override
   size_t MemoryLeft(); // override
   ~FMemoryStack2(); // override

   /// create a new stack of size 'nSize' bytes on the global heap.
   void Create( std::size_t nSize );
   /// create a new stack in the storage area marked by 'pBase',
   /// of 'nSize' bytes. This assumes that pBase is already allocated
   /// from elsewhere.
   void AssignMemory( char *pBase_, std::size_t nSize );
   void Destroy();

   //private:
   std::size_t
      m_Pos, m_Size;
   char *m_pData;

   bool m_bOwnMemory; // if true, we allocated m_pData and need to free it on Destroy().

private:
    FMemoryStack2(FMemoryStack2 const &); // not implemented
    void operator = (FMemoryStack2 const &); // not implemented
};



/// split a base memory stack into sub-stacks, one for each openmp thread.
struct FMemoryStackArray
{
    explicit FMemoryStackArray( FMemoryStack &BaseStack );
    ~FMemoryStackArray();

    FMemoryStack2
        *pSubStacks;
    FMemoryStack
        *pBaseStack;
    uint
        nThreads;

    // return substack for the calling openmp thread.
    FMemoryStack2 &GetStackOfThread();

    // destroy this object (explicitly) already now
    void Release();

    char
        *pStackBase;
private:
    FMemoryStackArray(FMemoryStackArray const &); // not implemented
    void operator = (FMemoryStackArray const &); // not implemented
};



} // namespace ct


#endif // _CX_MEMORYSTACK_H
