/*
  Developed by Sandeep Sharma
  Copyright (c) 2017, Sandeep Sharma
  
  This file is part of DICE.
  
  This program is free software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation, 
  either version 3 of the License, or (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with this program. 
  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef Global_HEADER_H
#define Global_HEADER_H
#include <boost/interprocess/managed_shared_memory.hpp>
#include <random>
#ifndef SERIAL
#include "mpi.h"
extern MPI_Comm shmcomm, localcomm;
#endif
extern int commrank, shmrank, localrank;
extern int commsize, shmsize, localsize;

#ifdef Complex
#define MatrixXx MatrixXcd
#define CItype std::complex<double>
#else
#define MatrixXx MatrixXd
#define CItype double
#endif

const int DetLen = 3;

extern boost::interprocess::shared_memory_object int2Segment;
extern boost::interprocess::mapped_region regionInt2;
extern std::string shciint2;

extern std::mt19937 generator;
double getTime();
void license();

#endif

