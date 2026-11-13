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
#ifndef HMULT_HEADER_H
#define HMULT_HEADER_H
#include <Eigen/Dense>
#include <Eigen/Core>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"
#include "global.h"
#include "Determinants.h"
#include <algorithm>
#include <chrono>
#include "SHCISortMpiUtils.h"
#include "SHCImake4cHamiltonian.h"

using namespace Eigen;
using namespace std;
using namespace SHCISortMpiUtils;
using namespace SHCImake4cHamiltonian;

std::complex<double> sumComplex(const std::complex<double>& a, const std::complex<double>& b) ;

namespace SHCISortMpiUtils{
  int ipow(int base, int exp);
};

struct HmultDirectFCI {
  Determinant* dets_ptr;
  size_t* dets_to_singles_ptr;
  int dets_size;
  int nelec;
  int norbs;
  oneInt & I1;
  twoInt & I2;
  double & coreE;
  MatrixXx & diag;

  HmultDirectFCI(
    Determinant* p_dets_ptr,
    size_t* p_dets_to_singles_ptr,
    int p_dets_size,
    int p_nelec,
    int p_norbs,
    oneInt& pI1,
    twoInt& pI2,
    double& pcoreE,
    MatrixXx & pDiag
  ) : 
    dets_ptr(p_dets_ptr),
    dets_to_singles_ptr(p_dets_to_singles_ptr),
    dets_size(p_dets_size),
    nelec(p_nelec),
    norbs(p_norbs),
    I1(pI1),
    I2(pI2),
    coreE(pcoreE),
    diag(pDiag) {};
  void operator()(CItype *x, CItype *y) {
    Determinant d0 = dets_ptr[0];
    int nopen = norbs - nelec;
    int nclose = nelec;
    int nsingles = nopen * nclose;

#ifndef SERIAL
    boost::mpi::communicator world;
#endif

    for (int i=0; i<dets_size; i++) {
      auto idet = i;
      Determinant det_i = dets_ptr[idet];
      y[idet] = det_i.Energy(I1, I2, coreE)*x[idet];
      for (int j=0; j < nsingles; j++) {
        int jdet = dets_to_singles_ptr[idet * nsingles + j];
        size_t orbDiff;
        Determinant det_j = dets_ptr[jdet];
        y[idet] += Hij(det_i, det_j, I1, I2, coreE, orbDiff)*x[jdet];
        for (int k=0; k < nsingles; k++) {
          int kdet = dets_to_singles_ptr[jdet * nsingles + k];
          Determinant det_k = dets_ptr[kdet];
          if(det_i.ExcitationDistance(det_k) <= 1) continue; 
          y[idet] += 0.25*Hij(det_i, det_k, I1, I2, coreE, orbDiff)*x[kdet];
        }
      }
    }
  }
};

struct Hmult2 {
  SparseHam &sparseHam;

  Hmult2(SparseHam& p_sparseHam) : sparseHam(p_sparseHam) {}

  //=============================================================================
  void operator()(CItype *x, CItype *y) {
    //-----------------------------------------------------------------------------
    /*!
    Calculate y = H.x from the sparse Hamiltonian
    :Inputs:
        CItype *x:
            The vector to multiply by H
        CItype *y:
            The vector H.x (output)
    */
    //-----------------------------------------------------------------------------

#ifndef SERIAL
    boost::mpi::communicator world;
#endif
    int size = commsize, rank = commrank;

    int numDets = sparseHam.connections.size(),
        localDets = sparseHam.connections.size();
#ifndef SERIAL
    MPI_Allreduce(&localDets, &numDets, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    // TODO
    auto start = std::chrono::system_clock::now();

    vector<CItype> ytemp(numDets, 0);
    int i, j;
#pragma omp declare reduction(vec_CItype_plus : std::vector<CItype> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<CItype>())) \
                    initializer(omp_priv = omp_orig)

#pragma omp parallel for private(i, j) reduction(vec_CItype_plus : ytemp)
    for (i = 0; i < sparseHam.connections.size(); i++) {
      for (j = 0; j < sparseHam.connections[i].size(); j++) {
        CItype hij = sparseHam.Helements[i][j];
        int J = sparseHam.connections[i][j];
        int I = i * size + rank;
	ytemp[J] += hij * x[I];
#ifdef Complex
        if (J != I) ytemp[I] += std::conj(hij) * x[J];
#else 
        if (J != I) ytemp[J] += hij * x[I];
#endif
      }
    }
    if (rank == 0) {
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
    }
    /*start = std::chrono::system_clock::now();
    ytemp = vector<CItype>(numDets, 0);
    for (int i = 0; i < sparseHam.connections.size(); i++) {
      for (int j = 0; j < sparseHam.connections[i].size(); j++) {
        CItype hij = sparseHam.Helements[i][j];
        int J = sparseHam.connections[i][j];
        int I = i * size + rank;

	ytemp[I] += hij * x[J];

        if (J != I) {
          ytemp[J] += hij * x[i * size + rank];
        }
      }
    }
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
      pout << "Time cost of Hmult on single thread " << elapsed_seconds.count() << endl;*/
// Reduce result vector y (if using distributed memory)
#ifndef SERIAL
#ifndef Complex
    if (localrank == 0) {
      MPI_Reduce(MPI_IN_PLACE, &ytemp[0], numDets, MPI_DOUBLE, MPI_SUM, 0,
                 localcomm);
      for (int j = 0; j < numDets; j++) y[j] = ytemp[j];
    } else {
      MPI_Reduce(&ytemp[0], &ytemp[0], numDets, MPI_DOUBLE, MPI_SUM, 0,
                 localcomm);
    }
#else
    if (localrank == 0) {
      MPI_Reduce(MPI_IN_PLACE, &ytemp[0], 2 * numDets, MPI_DOUBLE, MPI_SUM, 0,
                 localcomm);
      for (int j = 0; j < numDets; j++) y[j] = ytemp[j];
    } else {
      MPI_Reduce(&ytemp[0], &ytemp[0], 2 * numDets, MPI_DOUBLE, MPI_SUM, 0,
                 localcomm);
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
#else
    for (int j = 0; j < numDets; j++) y[j] = ytemp[j];
#endif
    // TODO
    if (rank == 0) {
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
    }
  }  // operator
};
#endif
