/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

#ifndef CX_FIXED_SIZE_ARRAY_H
#define CX_FIXED_SIZE_ARRAY_H

#include <functional> // for std::less

/// Array class with fixed maximum size which stores its elements
/// in-place (i.e., no allocations).
///
/// For technical reasons, all MaxN array elements are default-constructed on
/// construction and only destroyed when TArrayFix is (i.e., technically,
/// all array elements are alive all the time even if size() < max_size()).
template<class FType, unsigned int MaxN, class FSizeType = std::size_t>
struct TArrayFix
{
    typedef FType value_type;
    typedef FType *iterator;
    typedef FType const *const_iterator;
    typedef FSizeType size_type;
    typedef size_type size_t;

    // compiler-generated default destructor, copy-ctor and assignment op should work.
    TArrayFix()
        : nSize(0)
    {};

    explicit TArrayFix( size_t nEntries )
        : nSize(0)
    { resize(nEntries); };

    TArrayFix( size_t nEntries, FType const &Scalar )
        : nSize(0)
    { resize(nEntries); *this = Scalar; };


    template<class FInputIt>
    TArrayFix( FInputIt first, FInputIt last ){
        nSize = 0;
        while( first != last )
            push_back(*(first++));
    }

    FType &operator[] ( size_t i ){ assert(i < nSize); return m[i]; };
    FType const &operator[] ( size_t i ) const {  assert(i < nSize); return m[i]; };

    FType &back(){ assert(nSize!=0); return m[nSize-1]; };
    FType const &back() const { assert(nSize!=0); return m[nSize-1]; };
    FType &front(){ assert(nSize!=0); return m[0]; };
    FType const &front() const { assert(nSize!=0); return m[0]; };

    bool operator == ( TArrayFix const &other ) const {
        if ( size() != other.size() )
            return false;
        for ( size_t i = 0; i != size(); ++ i )
            if ( (*this)[i] != other[i] )
                return false;
        return true;
    }

    bool operator != ( TArrayFix const &other ) const {
        return !this->operator ==(other);
    }

    size_t size() const { return nSize; }
    bool empty() const { return nSize == 0; }
    size_t capacity() const { return MaxN; }
    void clear() { resize(0); };

    void push_back( FType const &t ){
        assert( nSize < MaxN );
        m[nSize] = t;
        ++nSize;
    }
    void pop_back(){
        assert( nSize > 0 );
        --nSize;
    }
    void resize( size_t NewSize ){
        assert( NewSize <= MaxN );
        nSize = NewSize;
    }
    void resize( size_t NewSize, FType const &value ){
        assert( NewSize <= MaxN );
        for ( size_t i = nSize; i < NewSize; ++ i )
            m[i] = value;
        nSize = NewSize;
    }

    iterator erase( iterator itFirst, iterator itLast ){
        assert( itFirst >= begin() && itLast <= end() && itFirst <= itLast );
        nSize -= itLast - itFirst;
        for ( iterator it = itFirst; it < end(); ++ it, ++ itLast )
            *it = *itLast;
        return itFirst;
    };

    iterator erase( iterator itWhere ){
        return erase(itWhere, itWhere+1);
    };

    FType *data() { return &m[0]; };
    FType const *data() const { return &m[0]; };

    // assign Scalar to every element of *this
    void operator = ( FType const &Scalar ){
        for ( iterator it = begin(); it != end(); ++ it )
            *it = Scalar;
    }

    template <class FIt>
    void assign(FIt first, FIt last) {
        resize(last - first);
        for ( FSizeType i = 0; i < nSize; ++ i )
            m[i] = first[i];
    }

    iterator begin(){ return &m[0]; }
    iterator end(){ return &m[nSize]; }
    const_iterator begin() const { return &m[0]; }
    const_iterator end() const { return &m[nSize]; }

protected:
    FType
        m[MaxN];
    FSizeType
        nSize; // actual number of elements (<= MaxN).
};

// lexicographically compare two arrays. -1: A < B; 0: A == B; +1: A > B.
template<class FType, unsigned int MaxN, class FPred>
int Compare(TArrayFix<FType,MaxN> const &A, TArrayFix<FType,MaxN> const &B, FPred less = std::less<FType>())
{
    if ( A.size() < B.size() ) return -1;
    if ( B.size() < A.size() ) return +1;
    for ( unsigned int i = 0; i < A.size(); ++ i ) {
        if ( less(A[i], B[i]) ) return -1;
        if ( less(B[i], A[i]) ) return +1;
    }
    return 0;
}


#endif // CX_FIXED_SIZE_ARRAY_H


// kate: space-indent on; tab-indent on; backspace-indent on; tab-width 4; indent-width 4; mixedindent off; indent-mode normal;
