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
#include <Eigen/Dense>
#include "input.h"
#ifndef SERIAL
#include "mpi.h"
extern MPI_Comm shmcomm, localcomm;
#endif
//class schedule;
class Profile;
class twoInt;
class oneInt;
class twoIntHeatBathSHM;

extern int commrank, shmrank, localrank;
extern int commsize, shmsize, localsize;

#ifdef Complex
#define MatrixXx MatrixXcd
#define CItype std::complex<double>
#else
#define MatrixXx Eigen::MatrixXd
#define CItype double
#endif

const int DetLen = 10;
const int innerDetLen = 2;

const int max_nreplicas = 8;

extern boost::interprocess::shared_memory_object int2Segment;
extern boost::interprocess::mapped_region regionInt2;
extern std::string shciint2;

extern boost::interprocess::shared_memory_object int2SHMSegment;
extern boost::interprocess::mapped_region regionInt2SHM;
extern std::string shciint2shm;

extern boost::interprocess::shared_memory_object int2SHMCASSegment;
extern boost::interprocess::mapped_region regionInt2SHMCAS;
extern std::string shciint2shmcas;

extern boost::interprocess::shared_memory_object cholSegment;
extern boost::interprocess::mapped_region cholRegion;
extern std::string cholSHMName;

extern boost::interprocess::shared_memory_object cholSegmentUp, cholSegmentDn;
extern boost::interprocess::mapped_region cholRegionUp, cholRegionDn;
extern std::string cholSHMNameUp, cholSHMNameDn;

extern boost::interprocess::shared_memory_object floatCholSegment;
extern boost::interprocess::mapped_region floatCholRegion;
extern std::string floatCholSHMName;

extern boost::interprocess::shared_memory_object rotCholSegment;
extern boost::interprocess::mapped_region rotCholRegion;
extern std::string rotCholSHMName;

extern std::mt19937 generator;
double getTime();
void   license();
extern schedule schd;
extern double startofCalc;

extern Profile prof;

extern twoInt I2;
extern oneInt I1;
extern double coreE;
extern twoIntHeatBathSHM I2hb;
extern twoIntHeatBathSHM I2hbCAS;
#endif
