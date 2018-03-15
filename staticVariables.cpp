#include "MoDeterminants.h"
#include "Determinants.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include "Eigen/Dense"
#include <string>
#ifndef SERIAL
#include "mpi.h"
#endif
int MoDeterminant::norbs = 1;
int MoDeterminant::nalpha = 1;
int MoDeterminant::nbeta = 1;

int HalfDet::norbs =1;
int Determinant::norbs = 1;
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0;

Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder ;

boost::interprocess::shared_memory_object int2Segment;
boost::interprocess::mapped_region regionInt2;
std::string shciint2;

#ifndef SERIAL
MPI_Comm shmcomm, localcomm;
#endif
int commrank, shmrank, localrank;
int commsize, shmsize, localsize;

