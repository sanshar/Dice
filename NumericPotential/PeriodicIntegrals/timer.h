/*
  Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012                      
  Copyright (c) 2012, Garnet K.-L. Chan                                        
  
  This program is integrated in Molpro with the permission of 
  Sandeep Sharma and Garnet K.-L. Chan
*/


#pragma once

#include <ctime>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <execinfo.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

class cumulTimer
{
 private:
  double localStart;
  double cumulativeSum;

 public:

  cumulTimer() : localStart(-1.0), cumulativeSum(0) {};

  void reset() {localStart = -1.0; cumulativeSum = 0.;}

  void start() {
    struct timeval start;
    gettimeofday(&start, NULL);
    localStart = start.tv_sec + 1.e-6*start.tv_usec;
  }



  void stop() 
  {

    struct timeval start;
    gettimeofday(&start, NULL);
    cumulativeSum = cumulativeSum + (start.tv_sec + 1.e-6*start.tv_usec) - localStart;
    if ((start.tv_sec + 1.e-6*start.tv_usec) - localStart < 0 ) 
    {
	  
      cout << "local stop called without starting first"<<endl;
      cout << localStart<<"  "<<(start.tv_sec + 1.e-6*start.tv_usec)<<endl;
      throw 20;
      assert(1==2);
      abort();
    }


    localStart = 0;

  }

  friend ostream& operator<<(ostream& os, const cumulTimer& t)
  {
    os << ((float)t.cumulativeSum);
    return os;
  }

};
