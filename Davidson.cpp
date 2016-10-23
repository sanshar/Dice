#include "Davidson.h"
#include "Hmult.h"
#include <Eigen/Dense>
#include <iostream>
#include <iostream>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"

using namespace Eigen;
using namespace std;


void precondition(MatrixXd& r, MatrixXd& diag, double& e) {
  for (int i=0; i<r.rows(); i++) {
    double eps = 1.e-8;
    if (abs(e-diag(i,0)) > 1e-8)
      r(i,0) = r(i,0)/(e-diag(i,0));
    else
      r(i,0) = r(i,0)/(e-diag(i,0)-1.e-8);
  }
}

//davidson, implemented very similarly to as implementeded in Block

double LinearSolver(Hmult2& H, MatrixXd& x0, MatrixXd& b, double tol, bool print) {
#ifndef SERIAL
    boost::mpi::communicator world;
#endif

  x0.setZero(x0.rows(),1);
  MatrixXd r = 1.*b, p = 1.*b;
  double rsold = (r.transpose()*r)(0,0);

  int iter = 0;
  while (true) {
    MatrixXd Ap = 0.*p; H(p,Ap);
    double alpha = rsold/(p.transpose()*Ap)(0,0);
    x0 += alpha*p;
    r -= alpha*Ap;

    double rsnew = (r.transpose()*r)(0,0);
    double ept = -(x0.transpose()*r + x0.transpose()*b)(0,0);
    if (true)
      pout <<"#"<< iter<<" "<<ept<<"  "<<rsnew<<std::endl;
    if (r.norm() < tol || iter > 100) { 
      p.setZero(p.rows(),1); H(x0,p); p -=b; cout << (p.transpose()*p)(0,0)<<endl; 
      return ept;
    }      

    p = r +(rsnew/rsold)*p;
    rsold = rsnew;
    iter++;
  }

  
}



//davidson, implemented very similarly to as implementeded in Block
double davidson(Hmult2& H, MatrixXd& x0, MatrixXd& diag, int maxCopies, double tol, bool print) {
#ifndef SERIAL
    boost::mpi::communicator world;
#endif
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
    MatrixXd hsubspace(bsize, bsize);hsubspace.setZero(bsize, bsize);
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
    if (true)
      pout <<"#"<< iter<<" "<<e0<<"  "<<error<<std::endl;
    iter++;
    if (error < tol || iter >200) {
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
      bsize = 6;
      sigmaSize = 5;
      b.col(bsize-1) = r/r.norm();
    }
  }
}



