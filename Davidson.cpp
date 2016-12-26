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
#include "boost/format.hpp"
#include "communicate.h"
#include "iowrapper.h"

using namespace Eigen;
using namespace std;
using namespace boost;

void precondition(MatrixXd& r, MatrixXd& diag, double& e) {
  for (int i=0; i<r.rows(); i++) {
    if (abs(e-diag(i,0)) > 1e-12)
      r(i,0) = r(i,0)/(e-diag(i,0));
    else
      r(i,0) = r(i,0)/(e-diag(i,0)-1.e-8);
  }
}

//davidson, implemented very similarly to as implementeded in Block
//davidson, implemented very similarly to as implementeded in Block
vector<double> davidson(Hmult2& H, vector<MatrixXd>& x0, MatrixXd& diag, int maxCopies, double tol, bool print) {
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  std::vector<double> eroots;

  int nroots = x0.size();
  MatrixXd b=MatrixXd::Zero(x0[0].rows(), maxCopies); 

  //if some vector has zero norm then randomise it
  if (mpigetrank() == 0) {
    for (int i=0; i<nroots; i++)  {
      b.col(i) = 1.*x0[i];
      if (x0[i].norm() < 1.e-10) {
	b.col(i).setRandom();
	b.col(i) = b.col(i)/b.col(i).norm();
      }
    }
  }
  mpi::broadcast(world, b, 0);

  //make vectors orthogonal to each other
  for (int i=0; i<x0.size(); i++) {  
    for (int j=0; j<i; j++) {
      double overlap = (b.col(i).transpose()*b.col(j))(0,0);
      b.col(i) -= overlap*b.col(j);
    }
    b.col(i) = b.col(i)/b.col(i).norm();
  }



  MatrixXd sigma = MatrixXd::Zero(x0[0].rows(), maxCopies); 
  int sigmaSize=0, bsize = x0.size();
  MatrixXd r(x0[0].rows(),1); r= 0.0*x0[0];
  int convergedRoot = 0;

  int iter = 0;

  while(true) {
    for (int i=sigmaSize; i<bsize; i++) {
      Eigen::Block<MatrixXd> bcol = b.block(0,i,b.rows(),1), sigmacol = sigma.block(0,i,sigma.rows(),1);
      H(bcol, sigmacol);
      sigmaSize++;
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

    double ei = eigensolver.eigenvalues()[convergedRoot];
    for (int i=0; i<convergedRoot; i++) {
      r = sigma.col(i) - eigensolver.eigenvalues()[i]*b.col(i);
      double error = r.norm();
      if (error > tol) {
	convergedRoot = i;
	pout << "going back to converged root "<<i<<endl;
	continue;
      }
    }

    r = sigma.col(convergedRoot) - ei*b.col(convergedRoot);
    double error = r.norm();

    if (false)
      pout <<"#"<< iter<<" "<<convergedRoot<<"  "<<ei<<"  "<<error<<std::endl;
    iter++;

    if (error < tol || iter >200) {
      if (iter >200) return eroots;
      convergedRoot++;
      pout << str(boost::format("#converged root:%3d -> Energy : %18.10g  \n") % (convergedRoot-1) % ei );
      if (convergedRoot == nroots) {
	for (int i=0; i<convergedRoot; i++) {
	  x0[i] = b.col(i);
	  eroots.push_back(eigensolver.eigenvalues()[i]);
	}
	return eroots;
      }
    }

    precondition(r,diag,ei);
    for (int i=0; i<bsize; i++) 
      r = r - (r.cwiseProduct(b.col(i)).sum())*b.col(i);
    //r = r - (b.col(i).transpose()*r)(0,0)*b.col(i);

    if (bsize < maxCopies) {
      b.col(bsize) = r/r.norm();
      bsize++;
    }
    else {
      bsize = nroots+3;
      sigmaSize = nroots+2;
      b.col(bsize-1) = r/r.norm();
    }
  }
}




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






