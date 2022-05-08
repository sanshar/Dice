
#include "Determinants.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include "Eigen/Dense"
#include <string>
#include <ctime>
#include <sys/time.h>
#include "input.h"
#include "Profile.h"
#include "integral.h"
#ifndef SERIAL
#include "mpi.h"
#endif


int Determinant::norbs = 1;
int Determinant::nalpha = 1;
int Determinant::nbeta = 1;
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0;

twoInt I2;
oneInt I1;
double coreE;
twoIntHeatBathSHM I2hb(1e-10);
twoIntHeatBathSHM I2hbCAS(1e-10);

Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder ;

boost::interprocess::shared_memory_object int2Segment;
boost::interprocess::mapped_region regionInt2;
std::string shciint2;

boost::interprocess::shared_memory_object int2SHMSegment;
boost::interprocess::mapped_region regionInt2SHM;
std::string shciint2shm;

boost::interprocess::shared_memory_object int2SHMCASSegment;
boost::interprocess::mapped_region regionInt2SHMCAS;
std::string shciint2shmcas;

boost::interprocess::shared_memory_object cholSegment;
boost::interprocess::mapped_region cholRegion;
std::string cholSHMName;

boost::interprocess::shared_memory_object cholSegmentUp, cholSegmentDn;
boost::interprocess::mapped_region cholRegionUp, cholRegionDn;
std::string cholSHMNameUp, cholSHMNameDn;

boost::interprocess::shared_memory_object floatCholSegment;
boost::interprocess::mapped_region floatCholRegion;
std::string floatCholSHMName;

boost::interprocess::shared_memory_object rotCholSegment;
boost::interprocess::mapped_region rotCholRegion;
std::string rotCholSHMName;

std::mt19937 generator;

#ifndef SERIAL
MPI_Comm shmcomm, localcomm;
#endif
int commrank, shmrank, localrank;
int commsize, shmsize, localsize;

schedule schd;
Profile prof;

double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc;

void license() {
  return;
  if (commrank == 0) {
  cout << endl;
  cout << endl;
  cout << "**************************************************************"<<endl;
  cout << "Dice  Copyright (C) 2017  Sandeep Sharma"<<endl;
  cout <<"This program is distributed in the hope that it will be useful,"<<endl;
  cout <<"but WITHOUT ANY WARRANTY; without even the implied warranty of"<<endl;
  cout <<"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl;
  cout <<"See the GNU General Public License for more details."<<endl;
  cout << endl<<endl;
  cout << "Author:       Sandeep Sharma"<<endl;
  cout << "Please visit our group page for up to date information on other projects"<<endl;
  cout << "http://www.colorado.edu/lab/sharmagroup/"<<endl;
  cout << "**************************************************************"<<endl;
  cout << endl;
  cout << endl;
  }
}


