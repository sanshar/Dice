#ifndef Global_HEADER_H
#define Global_HEADER_H
#include <ctime>
#include <sys/time.h>
#include <boost/interprocess/managed_shared_memory.hpp>

typedef unsigned short ushort;
const int DetLen = 6;
extern double startofCalc;
double getTime();

#ifdef Complex
#define MatrixXx MatrixXcd
#define CItype std::complex<double>
#else
#define MatrixXx MatrixXd 
#define CItype double
#endif

extern boost::interprocess::shared_memory_object int2Segment;
extern boost::interprocess::mapped_region regionInt2;
extern boost::interprocess::shared_memory_object int2SHMSegment;
extern boost::interprocess::mapped_region regionInt2SHM;

#endif
