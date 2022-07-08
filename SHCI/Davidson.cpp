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

std::complex<double> sumComplex(const std::complex<double>& a, const std::complex<double>& b) { return a+b; };



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
    if (abs(e-diag(i,0)) > 1e-12)
      r(i,0) = r(i,0)/(e-diag(i,0));
    else
      r(i,0) = r(i,0)/(e-diag(i,0)-1.e-12);
  }
} // end precondition



//=============================================================================
vector<double> davidson(Hmult2& H, vector<MatrixXx>& x0, MatrixXx& diag, int maxCopies, double tol, int& numIter, bool print) {
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

  int nroots = x0.size();
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
      MatrixXx hsubspace(bsize, bsize);hsubspace.setZero(bsize, bsize);
      for (int i=0; i<bsize; i++)
        for (int j=i; j<bsize; j++) {
          hsubspace(i,j) = b.col(i).dot(sigma.col(j));
#ifdef Complex
          hsubspace(j,i) = conj(hsubspace(i,j));
#else
          hsubspace(j,i) = hsubspace(i,j);
#endif
      }
      SelfAdjointEigenSolver<MatrixXx> eigensolver(hsubspace);
      if (eigensolver.info() != Success) {
        pout << "Eigenvalue solver unsuccessful."<<endl;
        pout << hsubspace<<endl;
        abort();
      }

      b.block(0,0,b.rows(), bsize) = b.block(0,0,b.rows(), bsize)*eigensolver.eigenvectors();
      sigma.block(0,0,b.rows(), bsize) = sigma.block(0,0,b.rows(), bsize)*eigensolver.eigenvectors();

      ei = eigensolver.eigenvalues()[convergedRoot];
      for (int i=0; i<convergedRoot; i++) {
        r = sigma.col(i) - eigensolver.eigenvalues()[i]*b.col(i);
        double error = r.norm();
        if (error > tol) {
          convergedRoot = i;
          if (print) pout << "going back to converged root "<<i<<endl;
          continue;
        }
      }

      r = sigma.col(convergedRoot) - ei*b.col(convergedRoot);
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
          x0[i] = b.col(i);
          eroots.push_back(eigensolver.eigenvalues()[i]);
          if (print ) pout << str(boost::format("#niter:%3d root:%3d -> Energy : %18.10g  \n") %(numIter) % (i) % eroots[i] );
        }
        continueOrReturn = 2;
        goto label1;
        //return eroots;
      }

      if (error < tol || numIter >800*x0.size()) {
        if (numIter >2000*x0.size()) {
          pout << str(boost::format("Davidson calculation did not converge for root %3d, #iter %5d\n") % (convergedRoot+1) % (numIter) );
          exit(0);
          continueOrReturn = 2;
          //return eroots;
        }
        convergedRoot++;
        if(print) pout << str(boost::format("#niter:%3d root:%3d -> Energy : %18.10g  \n") %(numIter) % (convergedRoot-1) % ei );
        if (convergedRoot == nroots) {
          for (int i=0; i<convergedRoot; i++) {
            x0[i] = b.col(i);
            eroots.push_back(eigensolver.eigenvalues()[i]);
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

      if (commrank == 0) {
        precondition(r,diag,ei);
        for (int i=0; i<bsize; i++)
          r = r - (b.col(i).adjoint()*r)(0,0)*b.col(i)/(b.col(i).adjoint()*b.col(i));

        if (bsize < maxCopies) {
          b.col(bsize) = r/r.norm();
          bsize++;
        } else {
          bsize = nroots+3;
          sigmaSize = nroots+2;
          b.col(bsize-1) = r/r.norm();
        }
      } // commrank=0
  } // while
} // end davidson



//=============================================================================
vector<double> davidsonDirect(HmultDirect& Hdirect, vector<MatrixXx>& x0, MatrixXx& diag, int maxCopies, double tol, int& numIter, bool print) {
//-----------------------------------------------------------------------------
    /*!
    Davidson, implemented very similarly to as implementeded in Block

    :Inputs:

        HmultDirect& Hdirect:
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

        type name:
            BM_description
    */
//-----------------------------------------------------------------------------
  std::vector<double> eroots;

  CItype* bcol, *sigmacol;
  AllocateSHM(x0, bcol, sigmacol);

  int nroots = x0.size();
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
    for (int i=0; i<nroots; i++)  {
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
      Hdirect(bcol, sigmacol);
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
      MatrixXx hsubspace(bsize, bsize);hsubspace.setZero(bsize, bsize);
      for (int i=0; i<bsize; i++)
        for (int j=i; j<bsize; j++) {
          hsubspace(i,j) = b.col(i).dot(sigma.col(j));
#ifdef Complex
          hsubspace(j,i) = conj(hsubspace(i,j));
#else
          hsubspace(j,i) = hsubspace(i,j);
#endif
      }
      SelfAdjointEigenSolver<MatrixXx> eigensolver(hsubspace);
      if (eigensolver.info() != Success) {
        pout << "Eigenvalue solver unsuccessful."<<endl;
        pout << hsubspace<<endl;
        abort();
      }

      b.block(0,0,b.rows(), bsize) = b.block(0,0,b.rows(), bsize)*eigensolver.eigenvectors();
      sigma.block(0,0,b.rows(), bsize) = sigma.block(0,0,b.rows(), bsize)*eigensolver.eigenvectors();

      ei = eigensolver.eigenvalues()[convergedRoot];
      for (int i=0; i<convergedRoot; i++) {
        r = sigma.col(i) - eigensolver.eigenvalues()[i]*b.col(i);
        double error = r.norm();
        if (error > tol) {
          convergedRoot = i;
          if (print) pout << "going back to converged root "<<i<<endl;
          continue;
        }
      }

      r = sigma.col(convergedRoot) - ei*b.col(convergedRoot);
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
          x0[i] = b.col(i);
          eroots.push_back(eigensolver.eigenvalues()[i]);
          if (print ) pout << str(boost::format("#niter:%3d root:%3d -> Energy : %18.10g  \n") %(numIter) % (i) % eroots[i] );
        }
        continueOrReturn = 2;
        goto label1;
        //return eroots;
      }

      if (error < tol || numIter >400*x0.size()) {
        if (numIter >400*x0.size()) {
          pout << str(boost::format("Davidson calculation did not converge for root %3d, #iter %5d\n") % (convergedRoot+1) % (numIter) );
          exit(0);
          continueOrReturn = 2;
          //return eroots;
        }
        convergedRoot++;
        if(print) pout << str(boost::format("#niter:%3d root:%3d -> Energy : %18.10g  \n") %(numIter) % (convergedRoot-1) % ei );
        if (convergedRoot == nroots) {
          for (int i=0; i<convergedRoot; i++) {
            x0[i] = b.col(i);
            eroots.push_back(eigensolver.eigenvalues()[i]);
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

      if (commrank == 0) {
        precondition(r,diag,ei);
        for (int i=0; i<bsize; i++)
          r = r - (b.col(i).adjoint()*r)(0,0)*b.col(i)/(b.col(i).adjoint()*b.col(i));

        if (bsize < maxCopies) {
          b.col(bsize) = r/r.norm();
          bsize++;
        } else {
          bsize = min(nroots+3, maxCopies);
          sigmaSize = bsize-1;
          b.col(bsize-1) = r/r.norm();
        }
      } // commrank=0
  } // while
} // end davidsonDirect



//=============================================================================
double LinearSolver(Hmult2& H, double E0, MatrixXx& x0, MatrixXx& b, vector<CItype*>& proj, double tol, bool print) {
//-----------------------------------------------------------------------------
    /*!
    Solve (H0-E0)*x0 = b
    where "proj" is used to keep the solution orthogonal

    :Inputs:

        Hmult2& H:
            The matrix H0
        double E0:
            The energy E0
        MatrixXx& x0:
            The unknown vector x0 (output)
        MatrixXx& b:
            The right vector b
        vector<CItype*>& proj:
            Projector to keep the solution orthogonal
        double tol:
            Tolerance
        bool print:
            Triggers printing out of messages
    */
//-----------------------------------------------------------------------------
  for (int i=0; i<proj.size(); i++) {
    CItype dotProduct = 0.0, norm=0.0;
    for (int j=0; j<b.rows(); j++) {
#ifdef Complex
      dotProduct += conj(proj[i][j])*b(j,0);
      norm += conj(proj[i][j])*proj[i][j];
#else
      dotProduct += proj[i][j]*b(j,0);
      norm += proj[i][j]*proj[i][j];
#endif
    }
    for (int j=0; j<b.rows(); j++)
      b(j,0) = b(j,0) - dotProduct*proj[i][j]/norm;
  }

  x0.setZero(b.rows(),1);
  MatrixXx r = 1.*b, p = 1.*b;
  double rsold = r.squaredNorm();

  if (fabs(r.norm()) < tol) return 0.0;

  int iter = 0;
  while (true) {
    MatrixXx Ap = 0.*p;
    H(&p(0,0), &Ap(0,0)); ///REPLACE THIS WITH SOMETHING
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &Ap(0,0), Ap.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    Ap = Ap - E0*p; //H0-E0
    CItype alpha = rsold/(p.adjoint()*Ap)(0,0);
    x0 += alpha*p;
    r -= alpha*Ap;

    for (int i=0; i<proj.size(); i++) {
      CItype dotProduct = 0.0, norm=0.0;
      for (int j=0; j<b.rows(); j++) {
#ifdef Complex
        dotProduct += conj(proj[i][j])*r(j,0);
        norm += conj(proj[i][j])*proj[i][j];
#else
        dotProduct += proj[i][j]*r(j,0);
        norm += proj[i][j]*proj[i][j];
#endif
      }
      for (int j=0; j<r.rows(); j++)
        r(j,0) = r(j,0) - dotProduct*proj[i][j]/norm;
    }

    //r = r - ((proj[i].adjoint()*r)(0,0))*proj[i]/((proj[i].adjoint()*proj[i])(0,0));
    //r = r- ((proj.adjoint()*r)(0,0))*proj/((proj.adjoint()*proj)(0,0));

    double rsnew = r.squaredNorm();
    CItype ept = -(x0.adjoint()*r + x0.adjoint()*b)(0,0);
    if (false)
      pout <<"#"<< iter<<" "<<ept<<"  "<<rsnew<<std::endl;
    if (r.norm() < tol || iter > 100) {
      p.setZero(p.rows(),1);
      H(&x0(0,0), &p(0,0)); ///REPLACE THIS WITH SOMETHING
      p -=b;
      return abs(ept);
    }

    p = r +(rsnew/rsold)*p;
    rsold = rsnew;
    iter++;
  } // while
} // end LinearSolver

