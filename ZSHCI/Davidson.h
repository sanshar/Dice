/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam
  A. Holmes, 2017 Copyright (c) 2017, Sandeep Sharma

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
#ifndef DAVIDSON_HEADER_H
#define DAVIDSON_HEADER_H
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "boost/format.hpp"

#include "SHCItime.h"
#include "communicate.h"
#include "iowrapper.h"
#include "global.h"


using namespace Eigen;
using namespace std;
using namespace boost;

void AllocateSHM(vector<MatrixXx>& x0, CItype* &bcol, CItype* &sigmacol);

void precondition(MatrixXx& r, MatrixXx& diag, double& e);

//=============================================================================
template <typename Hmult>
vector<double> davidson(Hmult& H, vector<MatrixXx>& x0, MatrixXx& diag, int maxCopies, double tol, int& numIter, bool print, int nroots=1) {
//-----------------------------------------------------------------------------
    /*!
    BM_description

    :Inputs:

        Hmult2& H:
            BM_description
        vector<MatrixXx>& x0:
            BM_description
        MatrixXx& diag:
            BM_description
        int maxCopies:
            BM_description
        double tol:
            BM_description
        int& numIter:
            BM_description
        bool print:
            BM_description

    :Returns:

        std::vector<double> eroots:
            BM_description
    */
//-----------------------------------------------------------------------------
  std::vector<double> eroots;

  CItype* bcol, *sigmacol;
  AllocateSHM(x0, bcol, sigmacol);

  //int nroots = x0.size();
  MatrixXx b;
  if (commrank == 0)
    b=MatrixXx::Zero(x0[0].rows(), maxCopies);

  int brows = x0[0].rows();
#ifndef SERIAL
  MPI_Bcast(&brows, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  int niter;
  //if some vector has zero norm then randomise it
  if (commrank == 0) {
    for (int i=0; i<nroots; i++) {
      b.col(i) = 1.*x0[i];
      if (x0[i].norm() < 1.e-10) {
        b.col(i).setRandom();
        b.col(i) = b.col(i)/b.col(i).norm();
      }
    }

    //make vectors orthogonal to each other
    for (int i=0; i<x0.size(); i++) {
      for (int j=0; j<i; j++) {
        CItype overlap = (b.col(j).adjoint()*b.col(i))(0,0);
        b.col(i) -= overlap*b.col(j);
      }
      if (b.col(i).norm() <1e-8) {
        cout << "b.col(" << i << ") is set to random" << endl;
        b.col(i).setRandom();
      }
      for (int j=0; j<i; j++) {
        CItype overlap = (b.col(j).adjoint()*b.col(i))(0,0);
        b.col(i) -= overlap*b.col(j);
      }
      b.col(i) = b.col(i)/b.col(i).norm();
    } // i
  } // commrank=0

  MatrixXx sigma;
  if (commrank == 0) sigma = MatrixXx::Zero(x0[0].rows(), maxCopies);

  int sigmaSize=0, bsize = x0.size();
  MatrixXx r;
  if (commrank == 0) {r=MatrixXx::Zero(x0[0].rows(),1);}
  int convergedRoot = 0;

  //int iter = 0;
  numIter = 0;
  double ei = 0.0;
  while(true) {
    //0->continue with the loop, 1 -> continue clause, 2 -> return
    int continueOrReturn = 0;
#ifndef SERIAL
    MPI_Bcast(&bsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sigmaSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    for (int i=sigmaSize; i<bsize; i++) {
      if (commrank==0) {
        for (int k=0; k<brows; k++) bcol[k] = b(k,i);
      }
      for (int k=0; k<brows; k++) sigmacol[k] = 0.0;

      //by default the MatrixXx is column major,
      //so all elements of bcol are contiguous
#ifndef SERIAL
      MPI_Barrier(MPI_COMM_WORLD);
#ifndef Complex
      MPI_Bcast(bcol, brows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
      MPI_Bcast(bcol, 2*brows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      H(bcol, sigmacol);
      sigmaSize++;

#ifndef SERIAL
      MPI_Barrier(MPI_COMM_WORLD);

      if (localrank == 0) {
#ifndef Complex
        if (commrank == 0)
          MPI_Reduce(MPI_IN_PLACE, sigmacol,  brows, MPI_DOUBLE, MPI_SUM, 0, shmcomm);
        else
          MPI_Reduce(sigmacol, sigmacol,  brows, MPI_DOUBLE, MPI_SUM, 0, shmcomm);
#else
        if (commrank == 0)
          MPI_Reduce(MPI_IN_PLACE, sigmacol,  2*brows, MPI_DOUBLE, MPI_SUM, 0, shmcomm);
        else
          MPI_Reduce(sigmacol, sigmacol,  2*brows, MPI_DOUBLE, MPI_SUM, 0, shmcomm);
#endif
      } // localrank=0
      MPI_Barrier(MPI_COMM_WORLD);
#endif

      if (commrank==0) {
        for (int k=0; k<brows; k++) sigma(k,i) = sigmacol[k];
      }

#ifndef SERIAL
      MPI_Barrier(MPI_COMM_WORLD);
#endif
    } // i


    if (commrank == 0) {
      cout << "bsize" << bsize << endl;
      MatrixXx hsubspace(bsize, bsize);
      hsubspace.setZero(bsize, bsize);
      for (int i=0; i<bsize; i++)
        for (int j=i; j<bsize; j++) {
          hsubspace(i,j) = b.col(i).dot(sigma.col(j));
          hsubspace(j,i) = conj(hsubspace(i,j));
      }
      SelfAdjointEigenSolver<MatrixXx> eigensolver(hsubspace);
      if (eigensolver.info() != Success) {
        cout << "Eigenvalue solver unsuccessful."<<endl;
        cout << hsubspace<<endl;
        abort();
      }

      // Don't transform b and sigma here - only get eigenvalues
      // Transformation will be done only during subspace reduction
      // This reduces numerical instability and computational cost

      ei = eigensolver.eigenvalues()[convergedRoot];
      
      // Compute residual in the original basis using Ritz vector
      MatrixXx ritz_vector = b.block(0,0,b.rows(), bsize) * eigensolver.eigenvectors().col(convergedRoot);
      MatrixXx sigma_ritz = sigma.block(0,0,b.rows(), bsize) * eigensolver.eigenvectors().col(convergedRoot);
      
      for (int i=0; i<convergedRoot; i++) {
        MatrixXx ritz_i = b.block(0,0,b.rows(), bsize) * eigensolver.eigenvectors().col(i);
        MatrixXx sigma_i = sigma.block(0,0,b.rows(), bsize) * eigensolver.eigenvectors().col(i);
        r = sigma_i - eigensolver.eigenvalues()[i]*ritz_i;
        double error = r.norm();
        if (error > tol) {
          convergedRoot = i;
          if (print) pout << "going back to converged root "<<i<<endl;
          // CRITICAL FIX: Use goto instead of continue to properly restart the main loop
          goto label1;
        }
      }
      if (bsize >= maxCopies) {
        // Subspace reduction: Transform vectors to eigenbasis before reduction
        pout << "subspace reduction at iter " << numIter << endl;
        b.block(0,0,b.rows(), bsize) = b.block(0,0,b.rows(), bsize)*eigensolver.eigenvectors();
        sigma.block(0,0,b.rows(), bsize) = sigma.block(0,0,b.rows(), bsize)*eigensolver.eigenvectors();
      }

      r = sigma_ritz - ei*ritz_vector;
      double error = r.norm();
      //if (numIter == 0)
      //if (print ) pout << str(boost::format("#niter:%3d root:%3d -> Energy : %18.10g  \n") %(numIter) % (convergedRoot-1) % ei );
      if (print) {
        if (numIter == 0) printf("nIter  Root               Energy                Error\n");
        if (commrank == 0) printf ("%5i  %4i   %18.10g   %18.10g  %10.2f\n", numIter, convergedRoot, ei, error, (getTime()-startofCalc));
      }
      numIter++;


      if (hsubspace.rows() == b.rows()) {
        //all root are available
        for (int i=0; i<x0.size(); i++) {
          x0[i] = b.block(0,0,b.rows(), bsize) * eigensolver.eigenvectors().col(i);
          eroots.push_back(eigensolver.eigenvalues()[i]);
          if (print ) pout << str(boost::format("#niter:%3d root:%3d -> Energy : %18.10g  \n") %(numIter) % (i) % eroots[i] );
        }
        continueOrReturn = 2;
        goto label1;
        //return eroots;
      }

      if (error < tol || numIter >400*x0.size()) {
        if (numIter >400*x0.size()) {
          cout << "Davidson calculation Didnt converge"<<endl;
          exit(0);
          continueOrReturn = 2;
          //return eroots;
        }
        convergedRoot++;
        if(print) pout << str(boost::format("#niter:%3d root:%3d -> Energy : %18.10g  \n") %(numIter) % (convergedRoot-1) % ei );
        if (convergedRoot == nroots) {
      	  pout << convergedRoot << " " << nroots << endl;
          for (int i=0; i<convergedRoot; i++) {
            x0[i] = b.block(0,0,b.rows(), bsize) * eigensolver.eigenvectors().col(i);
            eroots.push_back(eigensolver.eigenvalues()[i]);
	          pout << eroots[i] << " " << x0[i](0,0) << endl;
          }
          continueOrReturn = 2;
          goto label1;
          //return eroots;
        }
      } // cvg
    } // commrank=0

    label1:
#ifndef SERIAL
      MPI_Bcast(&continueOrReturn, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&numIter         , 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
      if (continueOrReturn == 2) return eroots;

      double error = r.norm();
      if (commrank == 0 && error > tol) {
        precondition(r,diag,ei);
        for (int i=0; i<bsize; i++) {
          r = r - (b.col(i).adjoint()*r)(0,0)*b.col(i)/(b.col(i).adjoint()*b.col(i));
        }
        if (bsize < maxCopies) {
          b.col(bsize) = r/r.norm();
          bsize++;
        } else {
          //int newSize = max(nroots, int(maxCopies/2));
          pout << "maxCopies" << maxCopies << " nroots " << nroots << endl;
          // keep the collapsed subspace strictly inside the allocated maxCopies columns
          int newSize = max(nroots, min(nroots+4, maxCopies-1));
          for (int i = newSize; i<maxCopies; i++) {
            b.col(i).setZero();
            sigma.col(i).setZero();
          }
          bsize = newSize;
          // CRITICAL FIX: sigmaSize should match the actual computed sigma vectors
          sigmaSize = newSize;
          b.col(bsize) = r/r.norm();
          bsize++;
        }
      } // commrank=0
  } // while
} // end davidson

//vector<double> davidson(Hmult2& H, vector<MatrixXx>& x0, MatrixXx& diag,
//                        int maxCopies, double tol, int& numIter, bool print);

#endif
