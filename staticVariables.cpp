
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
int HalfDet::norbs = Determinant::norbs;

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
  pout << "Adam A Holmes, "
       << "Ankit Mahajan, "
       << "Bastien Mussard, "
       << "Iliya Sabzevari, " << endl
       << "James E. T. Smith, "
       << "Xubo Wang" << endl
       << endl;
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

  pout << "User:             " << user << std::endl;
  pout << "Date:             " << date << std::endl;
  pout << "PID:              " << getpid() << std::endl;
  pout << "Path:             " << argv[0] << std::endl;
  pout << "Commit:           " << git_commit << std::endl;
  pout << "Branch:           " << git_branch << std::endl;
  pout << "Compilation Date: " __DATE__ << " " << __TIME__ << std::endl;
}

// PT message
void log_pt(schedule& schd) {
  pout << endl;
  pout << endl;
  pout << "**************************************************************"
       << endl;
  pout << "PERTURBATION THEORY STEP  " << endl;
  pout << "**************************************************************"
       << endl;
  if (schd.stochastic == true && schd.DoRDM) {
    schd.DoRDM = false;
    pout << "(We cannot perform PT RDM with stochastic PT. Disabling RDM.)"
         << endl
         << endl;
  }
}