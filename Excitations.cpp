/*
Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
Copyright (c) 2017, Sandeep Sharma

This file is part of DICE.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "Determinants.h"
#include "CIPSIbasics.h"
#include "integral.h"
#include <vector>
#include "math.h"
#include "Hmult.h"
#include <tuple>
#include <map>
#include "Davidson.h"
#include "boost/format.hpp"
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"

using namespace std;
using namespace Eigen;
using namespace boost;
int HalfDet::norbs = 1; //spin orbitals
int Determinant::norbs = 1; //spin orbitals
int Determinant::EffDetLen = 1;
double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc = getTime();

void main(int argc, char* argv[]) {
#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  {
    twoInt I2; oneInt I1; int nelec; int norbs; double coreE, eps;
    std::vector<int> irrep;
    readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE, irrep);
    Determinant::norbs = 2*norbs; //spin orbitals
    Determinant::EffDetLen = Determinant::norbs/64+1;
    cout << Determinant::norbs<<"  "<<Determinant::EffDetLen<<endl;
  }
  int iter; std::vector<Determinant> Dets, SortedDets;
  {
    char file [5000];
    sprintf (file, "%d-variational.bkp" , 0 );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);

    load >> iter >> Dets >> SortedDets;
    cout << file<<"  "<<Dets.size()<<endl;
    ifs.close();
  }
  std::vector<int> ExcitationBins(50,0);
  ExcitationBins[0] = 1;
  for (int i=1; i<Dets.size(); i++) {
    int d = Dets[0].ExcitationDistance(Dets[i]);
    if (d == 0) {
      cout << i<<endl;
      cout << Dets[0]<<endl;
      cout << Dets[i]<<endl;
      exit(0);
    }
    ExcitationBins[d]++;
  }

  for (int i=0; i<ExcitationBins.size(); i++)
    cout << i<<"  "<<ExcitationBins[i]<<endl;
}
