#include "optimizer.h"
#include "Wfn.h"
#include "Determinants.h"
#include "MoDeterminants.h"
#include "evaluateE.h"
#include "input.h"
#include "global.h"
#include "boost/format.hpp"
#include "iowrapper.h"

using namespace boost;

namespace optimizer {
  void read(string fname, int& iter, VectorXd& vec1, VectorXd& vec2) {
    if (commrank == 0) {
      std::ifstream ifs(fname.c_str(), std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> vec1 >> vec2 >> iter;
      ifs.close();
    }
  }

  void write(string fname, int& iter, VectorXd& vec1, VectorXd& vec2) {
    if (commrank == 0) {
      std::ofstream ofs(fname.c_str(), std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << vec1 << vec2 << iter;
      ofs.close();
    }
  }

  void rmsprop(CPSSlater& wave, oneInt& I1, twoInt& I2, double& coreE) {

    int norbs = MoDeterminant::norbs,
       nalpha = MoDeterminant::nalpha, 
       nbeta  = MoDeterminant::nbeta; 


    double E0=0., gradnorm = 1.e4, stddev = 1.e4;
    double rt = 1;

    Eigen::VectorXd prevGrad = Eigen::VectorXd::Zero(wave.getNumVariables());
    Eigen::VectorXd sumsqGrad = Eigen::VectorXd::Zero(wave.getNumVariables());
    
    double momentum, momentumdecay = schd.momentumDecay, decay = schd.decay;
    double lrt = schd.gradientFactor; int epoch = schd.learningEpoch;

    if (commrank == 0) 
      std::cout << format("  Iter           Energy ( std-dev)      Grad-norm    LRate    CorrL     Time\n");

    int iter=0;
    if (schd.restart) {
      read("optimizer.bkp", iter, prevGrad, sumsqGrad);
#ifndef SERIAL
      MPI_Bcast(&(iter),     1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(prevGrad[0]), prevGrad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(sumsqGrad[0]),prevGrad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }

    for (; iter<schd.maxIter && gradnorm>schd.tol; iter++) {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(wave.getNumVariables());
      
      momentum = schd.momentum*exp(-momentumdecay*iter);
      //Nestorov's trick
      VectorXd vars = VectorXd::Zero(wave.getNumVariables());wave.getVariables(vars);
      if (abs(momentum ) > 1.e-5)
	{
	  VectorXd tempvars = VectorXd::Zero(grad.rows());
	  wave.getVariables(tempvars);
	  tempvars -= momentum*prevGrad;
	  wave.updateVariables(tempvars);
	}
      
      
      int stochasticIter = min(schd.stochasticIter*10, schd.stochasticIter* (int)( pow(2, floor( (1+iter)/epoch)) +0.1) ); 
      if (schd.deterministic) {
	getGradient(wave, E0, nalpha, nbeta, norbs, I1, I2, coreE, grad);
	stddev = 0.0;
      }
      else {
	getStochasticGradient(wave, E0, stddev, nalpha, nbeta, norbs, I1, I2, coreE, grad, rt, stochasticIter, 0.5e-3);
      }
      lrt = max(schd.mingradientFactor, schd.gradientFactor/pow(2.0, floor( (1+iter)/epoch)));
      gradnorm = grad.squaredNorm();
      for (int i=0; i<grad.rows(); i++) {
	sumsqGrad(i) = decay*sumsqGrad(i)+ (1-decay)*grad(i)*grad(i);
	grad(i) = momentum*prevGrad(i)+lrt*grad(i)/sqrt(sumsqGrad(i)+1e-8);
	vars(i) -= grad(i);
      }
      prevGrad = grad;
      
      
      wave.updateVariables(vars);
      wave.normalizeAllCPS();
      
#ifndef SERIAL
      MPI_Bcast(&(grad[0]),     grad.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&(vars[0]),     vars.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
      wave.writeWave();
      write("optimizer.bkp", iter, prevGrad, sumsqGrad);
      if (commrank == 0)
	std::cout << format("%6i   %14.8f (%8.2e) %14.8f %8.1e %8.1f %8.2f\n") %iter 
	  % E0 % stddev %(grad.norm()) %(lrt) %(rt) %( (getTime()-startofCalc));

    }
  }
};  



