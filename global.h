#ifndef Global_HEADER_H
#define Global_HEADER_H
#include <ctime>
#include <sys/time.h>

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

#endif
