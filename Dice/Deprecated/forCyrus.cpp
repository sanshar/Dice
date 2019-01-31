/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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
#include "Dice/Utils/Determinants.h"
#include "Dice/SHCIbasics.h"
#include "Dice/Utils/integral.h"
#include <vector>
#include "math.h"
#include "Dice/Hmult.h"
#include <tuple>
#include <map>
#include "Dice/Davidson.h"
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
#include "Dice/Utils/communicate.h"

using namespace std;
using namespace Eigen;
using namespace boost;

int HalfDet::norbs = 1; //spin orbitals
int Determinant::norbs = 1; //spin orbitals
int Determinant::EffDetLen = 1;
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder ;
double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc = getTime();

boost::interprocess::shared_memory_object int2Segment(boost::interprocess::open_or_create, "SHCIint2", boost::interprocess::read_write);
boost::interprocess::mapped_region regionInt2;
boost::interprocess::shared_memory_object int2SHMSegment(boost::interprocess::open_or_create, "SHCIint2", boost::interprocess::read_write);
boost::interprocess::mapped_region regionInt2SHM;

int main(int argc, char* argv[]) {

#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE, eps;
  std::vector<int> irrep;
  readIntegrals("FCIDUMP", I2, I1, nelec, norbs, coreE, irrep);

  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals
  HalfDet::norbs = norbs; //spin orbitals
  Determinant::EffDetLen = norbs/64+1;
  Determinant::initLexicalOrder(nelec);
  if (Determinant::EffDetLen >DetLen) {
    cout << "change DetLen in global.h to "<<Determinant::EffDetLen<<" and recompile "<<endl;
    exit(0);
  }

  char file [5000];
  sprintf (file, "%d-variational.bkp" , 0 );
  std::ifstream ifs(file, std::ios::binary);
  boost::archive::binary_iarchive load(ifs);

  int iter; std::vector<Determinant> Dets;
  load >> iter >> Dets ;
  {
    std::vector<Determinant> SortedDets;
    load >>SortedDets;
  }
  int diaglen;
  load >>diaglen;

  vector<MatrixXd> ci; double diag;
  ci.resize(1, MatrixXd(diaglen,1));

  for (int i=0; i<diaglen; i++)
    load >> diag;
  cout << diag<<endl;
  load >>ci;
  cout << "Read ci"<<endl;

  ifs.close();

  std::vector<int> ExcitationBins(50,0);
  std::vector<double> weights(50,0.);
  cout << Dets[0]<<endl;
  for (int i=0; i<Dets.size(); i++) {
    int d = Dets[0].ExcitationDistance(Dets[i]);
    if (d==0)
      cout <<" iter: "<<i<<" : "<< Dets[i]<<endl;
    ExcitationBins[d]++;
    weights[d] += ci[0](i,0)*ci[0](i,0);
  }

  for (int i=0; i<ExcitationBins.size(); i++)
    cout << i<<"  "<<ExcitationBins[i]<<"  "<<weights[i]<<endl;
  return 0;
}
