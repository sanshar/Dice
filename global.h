/*
Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef Global_HEADER_H
#define Global_HEADER_H
#include <ctime>
#include <sys/time.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <string>
#ifndef SERIAL
#include "mpi.h"
#endif

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
extern boost::interprocess::shared_memory_object hHelpersSegment;
extern boost::interprocess::mapped_region regionHelpers;
extern std::string shciHelper;

extern boost::interprocess::shared_memory_object DetsCISegment;
extern boost::interprocess::mapped_region regionDetsCI;
extern std::string shciDetsCI;

extern boost::interprocess::shared_memory_object DavidsonSegment;
extern boost::interprocess::mapped_region regionDavidson;
extern std::string shciDavidson;

extern boost::interprocess::shared_memory_object cMaxSegment;
extern boost::interprocess::mapped_region regioncMax;
extern std::string shcicMax;

extern MPI_Comm shmcomm;

#endif
