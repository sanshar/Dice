#ifndef DAVIDSON_HEADER_H
#define DAVIDSON_HEADER_H
#include <Eigen/Dense>
#include <Eigen/Core>
#include "Davidson.h"
#include "Hmult.h"
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;




void precondition(MatrixXd& r, MatrixXd& diag, double& e);

//davidson, implemented very similarly to as implementeded in Block
template<class T>
double davidson(T& H, MatrixXd& x0, MatrixXd& diag, int maxCopies, double tol, bool print) {
  x0 = x0/x0.norm();
  MatrixXd b(x0.rows(), maxCopies); b*=0.0;
  b.col(0) = 1.0*x0;
  MatrixXd sigma(x0.rows(), maxCopies); sigma*=0.0;
  int sigmaSize=0, bsize = 1;
  MatrixXd r(x0.rows(),1); r= 0.0*x0;

  int iter = 0;

  while(true) {
    for (int i=sigmaSize; i<bsize; i++) {
      Eigen::Block<MatrixXd> bcol = b.block(0,i,b.rows(),1), sigmacol = sigma.block(0,i,sigma.rows(),1);
      H(bcol, sigmacol);
    }
    MatrixXd hsubspace(bsize, bsize);hsubspace *= 0.;
    for (int i=0; i<bsize; i++)
      for (int j=i; j<bsize; j++) {
	hsubspace(i,j) = b.col(i).dot(sigma.col(j)); 
	hsubspace(j,i) = hsubspace(i,j);
      }

    SelfAdjointEigenSolver<MatrixXd> eigensolver(hsubspace);
    if (eigensolver.info() != Success) abort();

    b.block(0,0,b.rows(), bsize) = b.block(0,0,b.rows(), bsize)*eigensolver.eigenvectors();
    sigma.block(0,0,b.rows(), bsize) = sigma.block(0,0,b.rows(), bsize)*eigensolver.eigenvectors();

    double e0 = eigensolver.eigenvalues()[0];
    r = sigma.col(0) - e0*b.col(0);
    double error = r.norm();
    if (print)
      std::cout <<"#"<< iter<<" "<<e0<<"  "<<error<<std::endl;
    iter++;
    if (error < tol || iter >20) {
      x0 = 1.*b.col(0);
      return e0;
    }


    precondition(r,diag,e0);
    for (int i=0; i<bsize; i++) 
      r = r - (r.cwiseProduct(b.col(i)).sum())*b.col(i);

    if (bsize < maxCopies) {
      b.col(bsize) = r/r.norm();
      bsize++;
      sigmaSize++;
    }
    else {
      bsize = 3;
      sigmaSize = 2;
      b.col(bsize-1) = r/r.norm();
    }
  }
}

#endif
