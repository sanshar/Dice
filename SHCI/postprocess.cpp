/*
  Developed by Sandeep Sharma
  with contributions from James E. T. Smith and Adam A. Holmes
  2017 Copyright (c) 2017, Sandeep Sharma

  This file is part of DICE.

  This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <fstream>
#include <list>
#include <set>
#include <tuple>

#include "Davidson.h"
#include "Determinants.h"
#include "Hmult.h"
#include "SHCIbasics.h"
#include "SHCIgetdeterminants.h"
#include "SHCImakeHamiltonian.h"
#include "SHCIrdm.h"
#include "SHCItime.h"
#include "boost/format.hpp"
#include "global.h"
#include "input.h"
#include "integral.h"

#ifndef SERIAL
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#endif
#include <unistd.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/serialization/vector.hpp>
#include <cstdlib>
#include <numeric>
#ifndef Complex
#include "cdfci.h"
#endif
#include "LCC.h"
#include "SHCIshm.h"
#include "SOChelper.h"
#include "communicate.h"
#include "symmetry.h"
MatrixXd symmetry::product_table;
#include <algorithm>
#include "postprocess.h"

// Initialize
using namespace Eigen;
using namespace boost;
int HalfDet::norbs = 1;      // spin orbitals
int Determinant::norbs = 1;  // spin orbitals
int Determinant::EffDetLen = 1;
char Determinant::Trev = 0;  // Time reversal
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder;

// Get the current time
double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6 * start.tv_usec;
}
double startofCalc = getTime();

void readStates(char* file1, vector<MatrixXx>& ci1, vector<double>& E01, vector<Determinant>& Dets1)
{
  std::ifstream ifs(file1, std::ios::binary);
  boost::archive::binary_iarchive load(ifs);

  int iter;
  bool converged1;

  load >> iter >> Dets1;  // >>sorted ;
  load >> ci1;
  load >> E01;
  load >> converged1;

  ifs.close();
}

void mergeDeterminantsAndStates(vector<MatrixXx>& ci1, vector<MatrixXx>& ci2, vector<Determinant>& Dets1, vector<Determinant>& Dets2,
                                vector<MatrixXx>& mergedCi, vector<Determinant>& mergedDet)
{
  std::map<Determinant, std::pair<int, int>> mergedDetsMap;
  std::map<Determinant, std::pair<int, int>>::iterator it;

  for (int i=0; i<Dets1.size(); i++) {
    mergedDetsMap[Dets1[i]] = std::pair<int,int>(i, -1);
  }

  for (int i=0; i<Dets2.size(); i++) {
    it  =   mergedDetsMap.find(Dets2[i]);
    if (it == mergedDetsMap.end())
      mergedDetsMap[Dets2[i]] = std::pair<int, int>(-1, i);
    else
      it->second.second = i;
  }


  mergedDet.resize(mergedDetsMap.size());
  mergedCi.resize(ci1.size()+ci2.size(), MatrixXx::Zero(mergedDetsMap.size(),1));

  //cout << mergedDet.size()<<"  "<<mergedCi.size()<<"  "<<mergedCi[0].size()<<endl;

  int i=0;
  for (it=mergedDetsMap.begin(); it!=mergedDetsMap.end(); ++it) 
  {
    mergedDet[i] = it->first;

    for (int j=0; j<ci1.size(); j++) {
      if (it->second.first != -1)
        mergedCi[j](i,0) = ci1[j](it->second.first,0);
    }

    for (int j=0; j<ci2.size(); j++)
      if (it->second.second != -1)
        mergedCi[j+ci1.size()](i,0) = ci2[j](it->second.second,0);
    i++;
  }

}

void transitionRDMc(char * fname1, char* fname2, int norbs, int nelec)
{

  norbs = 2*norbs;
  Determinant::norbs = norbs;
  Determinant::Trev = 0;
  HalfDet::norbs = norbs;      // spin orbitals
  Determinant::EffDetLen = norbs / 64 + 1;
  Determinant::initLexicalOrder(nelec);

  int proc = 0, nprocs = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  boost::mpi::communicator world;
#endif
  commrank = proc; commsize = nprocs;

  initSHM();

  char file1[5000], file2[5000];
  sprintf(file1, "%s/%d-variational.bkp", fname1, commrank);
  sprintf(file2, "%s/%d-variational.bkp", fname2, commrank);

  vector<MatrixXx> ci1, ci2; 
  vector<double> E01, E02; 
  vector<Determinant> Dets1, Dets2;

  readStates(file1, ci1, E01, Dets1);
  cout << "Read "<<ci1.size()<<" states from file 1: "<<file1<<endl;
  
  readStates(file2, ci2, E02, Dets2);
  cout << "Read "<<ci2.size()<<" states from file 2: "<<file2<<endl;

  vector<MatrixXx> ci; vector<Determinant> Dets;
  mergeDeterminantsAndStates(ci1, ci2, Dets1, Dets2, ci, Dets);

  Determinant *SHMDets;
  SHMVecFromVecs(Dets, SHMDets, shciSortedDets, SortedDetsSegment, regionSortedDets);

  int DetsSize = Dets.size();
  Dets.clear();

#ifndef SERIAL
  mpi::broadcast(world, DetsSize, 0);
#endif

  int nroots = ci.size();
  SHCImakeHamiltonian::HamHelpers2 helpers2;
  SHCImakeHamiltonian::SparseHam sparseHam;

  if (proc == 0) {
    helpers2.PopulateHelpers(SHMDets, DetsSize, 0);
  }
  helpers2.MakeSHMHelpers();

  //make the connections
  SHCImakeHamiltonian::MakeConnectionsfromSMHelpers2(
      helpers2.AlphaMajorToBetaLen, helpers2.AlphaMajorToBetaSM,
      helpers2.AlphaMajorToDetSM, helpers2.BetaMajorToAlphaLen,
      helpers2.BetaMajorToAlphaSM, helpers2.BetaMajorToDetSM,
      helpers2.SinglesFromAlphaLen, helpers2.SinglesFromAlphaSM,
      helpers2.SinglesFromBetaLen, helpers2.SinglesFromBetaSM, SHMDets,
      0, DetsSize, false, sparseHam, norbs, true);


  //make a dummy schedule
  schedule schd;
  schd.DoSpinRDM = false;
  schd.prefix.push_back(".");
  assert(nelec == SHMDets[0].Noccupied());

  pout << "\nCalculating 2-RDM" << endl;

  MatrixXx twoRDM;// = MatrixXx::Zero(norbs * (norbs + 1) / 2,
                    //                norbs * (norbs + 1) / 2);
  MatrixXx s2RDM =
      MatrixXx::Zero((norbs / 2) * norbs / 2, (norbs / 2) * norbs / 2);


  for (int i = 0; i < nroots; i++) {
    CItype *SHMci;
    SHMVecFromMatrix(ci[i], SHMci, shciDetsCI, DetsCISegment,
                      regionDetsCI);

    for (int j=0; j<i+1; j++) {
 
      CItype *SHMci_j;
      SHMVecFromMatrix(ci[j], SHMci_j, shciDetsCI2, DetsCISegment2,
                      regionDetsCI2);

      s2RDM.setZero();
      SHCIrdm::EvaluateRDM(sparseHam.connections, SHMDets, DetsSize,
                          &ci[i](0,0), &ci[j](0,0), sparseHam.orbDifference, nelec,
                          schd, i, twoRDM, s2RDM);

      MatrixXx oneRDM = MatrixXx::Zero(norbs, norbs);
      MatrixXx s1RDM = MatrixXx::Zero(norbs / 2, norbs / 2);
      SHCIrdm::EvaluateOneRDM(sparseHam.connections, SHMDets, DetsSize,
                              SHMci, SHMci_j, sparseHam.orbDifference, nelec,
                              schd, i, oneRDM, s1RDM);
      SHCIrdm::save1RDM(schd, s1RDM, oneRDM, i, j);

      SHCIrdm::saveRDM(schd, s2RDM, twoRDM, i, j);

      //boost::interprocess::shared_memory_object::remove(
      //shciDetsCI2.c_str());
    }

    //boost::interprocess::shared_memory_object::remove(
    //shciDetsCI.c_str());
  }  // for i
}

