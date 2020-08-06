#include "sr.h"
#include <Eigen/Dense>
#include "global.h"
#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;
using namespace boost;


void ConjGrad(DirectMetric &A, VectorXd &b, int n, VectorXd &x)
{
  double tol = 1.e-10;
  
  VectorXd Ap = VectorXd::Zero(x.rows());
  A.multiply(x, Ap);
  VectorXd r = b - Ap;
  VectorXd p = r;
  
  double rsold = r.adjoint() * r;
  if (fabs(rsold) < tol) return;
  
  for (int i = 0; i < n; i++)
  {
    A.multiply(p, Ap);
    double pAp = p.adjoint() * Ap;
    double alpha = rsold / pAp;

    x = x + alpha * p;
    r = r - alpha * Ap;
    
    double rsnew = r.adjoint() * r;
    double beta = rsnew / rsold;

    rsold = rsnew;
    p = r + beta*p;
  }
}

void PInv(MatrixXd &A, MatrixXd &Ainv)
{
    SelfAdjointEigenSolver<MatrixXd> es;
    es.compute(A);
    std::vector<int> cols;
    for (int m = 0; m < A.cols(); m++)
    {
      if (fabs(es.eigenvalues()(m)) > 1.e-10)
      {
        cols.push_back(m);
      }
    }
    MatrixXd U = MatrixXd::Zero(A.rows(), cols.size());
    MatrixXd eig_inv = MatrixXd::Zero(cols.size(),cols.size());
    for (int m = 0; m < cols.size(); m++)
    {
      int index = cols[m];
      U.col(m) = es.eigenvectors().col(index);
      double eigval = es.eigenvalues()(index);
      eig_inv(m,m) = 1.0 / eigval;
    }
    Ainv = U * eig_inv * U.adjoint();
}

