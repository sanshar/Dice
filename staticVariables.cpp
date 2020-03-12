
#include <sys/time.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <ctime>
#include <string>
#include <vector>
#include "Determinants.h"
#include "Eigen/Dense"
#include "Profile.h"
#include "input.h"
#include "integral.h"
#ifndef SERIAL
#include "mpi.h"
#endif
#include <random>
#include "communicate.h"

int Determinant::norbs = 1;
int Determinant::n_spinorbs = Determinant::norbs * 2;
int Determinant::nalpha = 1;
int Determinant::nbeta = 1;
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0;
std::vector<int> irrep;

twoInt I2;
oneInt I1;
double coreE;
twoIntHeatBathSHM I2hb(1e-10);
twoIntHeatBathSHM I2hbCAS(1e-10);

Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder;

boost::interprocess::shared_memory_object int2Segment;
boost::interprocess::mapped_region regionInt2;
std::string shciint2;

boost::interprocess::shared_memory_object int2SHMSegment;
boost::interprocess::mapped_region regionInt2SHM;
std::string shciint2shm;

boost::interprocess::shared_memory_object int2SHMCASSegment;
boost::interprocess::mapped_region regionInt2SHMCAS;
std::string shciint2shmcas;

boost::interprocess::shared_memory_object hHelpersSegment;
boost::interprocess::mapped_region regionHelpers;
std::string shciHelper;

boost::interprocess::shared_memory_object DetsCISegment;
boost::interprocess::mapped_region regionDetsCI;
std::string shciDetsCI;

boost::interprocess::shared_memory_object SortedDetsSegment;
boost::interprocess::mapped_region regionSortedDets;
std::string shciSortedDets;

boost::interprocess::shared_memory_object DavidsonSegment;
boost::interprocess::mapped_region regionDavidson;
std::string shciDavidson;

boost::interprocess::shared_memory_object cMaxSegment;
boost::interprocess::mapped_region regioncMax;
std::string shcicMax;

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
  return start.tv_sec + 1.e-6 * start.tv_usec;
}
double startofCalc;

void license(char* argv[]) {
  // return;
  // if (commrank == 0) {
  //   cout << endl;
  //   cout << endl;
  //   cout << "**************************************************************"
  //        << endl;
  //   cout << "Dice  Copyright (C) 2017  Sandeep Sharma" << endl;
  //   cout << "This program is distributed in the hope that it will be useful,"
  //        << endl;
  //   cout << "but WITHOUT ANY WARRANTY; without even the implied warranty of"
  //        << endl;
  //   cout << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE." << endl;
  //   cout << "See the GNU General Public License for more details." << endl;
  //   cout << endl << endl;
  //   cout << "Author:       Sandeep Sharma" << endl;
  //   cout << "Please visit our group page for up to date information on other
  //   "
  //           "projects"
  //        << endl;
  //   cout << "http://www.colorado.edu/lab/sharmagroup/" << endl;
  //   cout << "**************************************************************"
  //        << endl;
  //   cout << endl;
  //   cout << endl;
  // }
  // void license(char* argv[]) {
  pout << endl;
  pout << "     ____  _\n";
  pout << "    |  _ \\(_) ___ ___\n";
  pout << "    | | | | |/ __/ _ \\\n";
  pout << "    | |_| | | (_|  __/\n";
  pout << "    |____/|_|\\___\\___|   v1.0\n";
  pout << endl;
  pout << endl;
  pout << "**************************************************************"
       << endl;
  pout << "Dice Copyright (C) 2020 Sandeep Sharma" << endl;
  pout << endl;
  pout << "This program is distributed in the hope that it will be useful,"
       << endl;
  pout << "but WITHOUT ANY WARRANTY; without even the implied warranty of"
       << endl;
  pout << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE." << endl;
  pout << "See the GNU General Public License for more details." << endl;
  pout << endl;
  pout << "Lead Author: Sandeep Sharma" << endl;
  pout << "Contributors (alphabetical): " << endl << endl;
  pout << "Adam A Holmes" << endl;
  pout << "Ankit Mahajan" << endl;
  pout << "Bastien Mussard" << endl;
  pout << "Iliya Sabzevari" << endl;
  pout << "James E. T. Smith" << endl;
  pout << "Xubo Wang" << endl << endl;
  pout << "For detailed documentation on Dice please visit" << endl;
  pout << "https://sanshar.github.io/Dice/" << endl;
  pout << "and our group page for up to date information on other projects"
       << endl;
  pout << "http://www.colorado.edu/lab/sharmagroup/" << endl;
  pout << "**************************************************************"
       << endl;
  pout << endl;

  char* user;
  user = (char*)malloc(10 * sizeof(char));
  user = getlogin();

  time_t t = time(NULL);
  struct tm* tm = localtime(&t);
  char date[64];
  strftime(date, sizeof(date), "%c", tm);

  printf("User:             %s\n", user);
  printf("Date:             %s\n", date);
  printf("PID:              %d\n", getpid());
  pout << endl;
  printf("Path:             %s\n", argv[0]);
  printf("Commit:           %s\n", git_commit);
  printf("Branch:           %s\n", git_branch);
  printf("Compilation Date: %s %s\n", __DATE__, __TIME__);
  // printf("Cores:            %s\n","TODO");
  // }
}
