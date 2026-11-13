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
#include "Davidson.h"
#include "Hmult.h"
#include "HmultDirect.h"
#include <Eigen/Dense>
#include <iostream>
#include <iostream>
#include "boost/format.hpp"
#include "communicate.h"
#include "iowrapper.h"
#include "global.h"

using namespace Eigen;
using namespace std;
using namespace boost;


//=============================================================================
void AllocateSHM(vector<MatrixXx>& x0, CItype* &bcol, CItype* &sigmacol){
//-----------------------------------------------------------------------------
    /*!
    Segment in shared memory

    :Inputs:

        vector<MatrixXx>& x0:
            BM_description
        CItype* &bcol:
            BM_description
        CItype* &sigmacol:
           BM_description
    */
//-----------------------------------------------------------------------------
  size_t totalMemory = 0, xrows=0;
  int comm_rank=0, comm_size=1;
#ifndef SERIAL
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif

  if (comm_rank == 0) {
    totalMemory = x0[0].rows()*2*(sizeof(CItype));
    xrows = x0[0].rows();
  }
#ifndef SERIAL
  MPI_Bcast(&totalMemory, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&xrows, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  DavidsonSegment.truncate(totalMemory);
  regionDavidson = boost::interprocess::mapped_region{DavidsonSegment, boost::interprocess::read_write};
  memset(regionDavidson.get_address(), 0., totalMemory);

#ifndef SERIAL
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  bcol = static_cast<CItype*>(regionDavidson.get_address());
  sigmacol = bcol + xrows;
  boost::interprocess::shared_memory_object::remove(shciDetsCI.c_str());
  boost::interprocess::shared_memory_object::remove(shciDavidson.c_str());
} // end AllocateSHM



//=============================================================================
void precondition(MatrixXx& r, MatrixXx& diag, double& e) {
//-----------------------------------------------------------------------------
    /*!
    Properly precondition the matrix "r"

    :Inputs:

        MatrixXx& r:
            Input/Ouput matrix to be preconditionned (output)
        MatrixXx& diag:
            Diagonal vector
        double& e:
            Threshold and shift
    */
//-----------------------------------------------------------------------------
  for (int i=0; i<r.rows(); i++) {
    double denominator = e - diag(i,0).real();
    if (abs(denominator) > 1e-5) {
      r(i,0) = r(i,0) / denominator;
    } else {
      // Avoid division by very small numbers
      r(i,0) = r(i,0) / double(1e-5);
    }
  }
} // end precondition
