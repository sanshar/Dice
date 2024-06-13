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

void readState(char * fname)
{
    /*
  int proc = 0, nprocs = 1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  boost::mpi::communicator world;
#endif
    */
   cout << "in here"<<endl;
   cout << fname<<endl;
    printf(fname);

    /*
    std::ifstream ifs(fname, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);

    int iter;
    vector<MatrixXx> ci;
    vector<double> E0;
    vector<Determinant> Dets;
    bool converged;

    load >> iter >> Dets;  // >>sorted ;
    load >> ci;
    load >> E0;
    load >> converged;

    ifs.close();

    cout << E0[0]<<endl;
    */
}

void readStatec(char * fname)
{
    readState(fname);
}
