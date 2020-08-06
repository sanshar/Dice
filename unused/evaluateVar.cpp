#include <vector>
#include <Eigen/Dense>
#include "Wfn.h"
#include <algorithm>
#include "integral.h"
#include "Determinants.h"
#include "Walker.h"
#include <boost/format.hpp>
#include <iostream>
#include "evaluateE.h"
#include "Davidson.h"
#include "Hmult.h"
#include <math.h>
#include "global.h"
#include "input.h"

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;
using namespace Eigen;

/*
void getVarianceGradiantDeterministic(CPSSlater &w, double &E0, int &nalpha, int &nbeta, int &norbs,
                              oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
                              VectorXd &grad, double &var)
{
  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;
  vector<double> HijElements;
  int nExcitations;

  double Overlap = 0, Energy = 0;
  grad.setZero();
  VectorXd diagonalGrad = VectorXd::Zero(grad.rows());
  VectorXd localdiagonalGrad = VectorXd::Zero(grad.rows());
  VectorXd localgrad = VectorXd::Zero(grad.rows());

  double E2 = 0.;
  VectorXd gradone = VectorXd::Zero(grad.rows());
  VectorXd gradtwo = VectorXd::Zero(grad.rows());
  VectorXd gradthree = VectorXd::Zero(grad.rows());
  VectorXd gradfour = VectorXd::Zero(grad.rows());

  for (int i = commrank; i < allDets.size(); i += commsize)
  {
    Walker walk(allDets[i]);
    walk.initUsingWave(w);
    double ovlp = 0, ham = 0;

    {
      E0 = 0.;
      double scale = 1.0;
      localgrad.setZero();
      localdiagonalGrad.setZero();
      gradone.setZero();
      gradtwo.setZero();
      gradthree.setZero();
      gradfour.setZero();

      w.HamAndOvlpGradient(walk, ovlp, ham, localgrad, I1, I2, I2hb, coreE, ovlpRatio,
                           excitation1, excitation2, HijElements, nExcitations, true, true);

      double tmpovlp = 1.0;
      w.OverlapWithGradient(walk, tmpovlp, localdiagonalGrad);
      double detovlp = walk.getDetOverlap(w);
      for (int k=0; k<w.ciExpansion.size(); k++) {
        localdiagonalGrad(k+w.getNumJastrowVariables()) += walk.alphaDet[k]*walk.betaDet[k]/detovlp;
       
      }
      if (w.determinants.size() <= 1 && schd.optimizeOrbs) {
        walk.OverlapWithGradient(w, localdiagonalGrad, detovlp);
      }
    }
*/
       
void getStochasticVarianceGradientContinuousTime(CPSSlater &w, double &E0, double &stddev,
						 int &nalpha, int &nbeta, int &norbs, oneInt &I1,
						 twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
						 VectorXd &grad, double &var, double &rk, int niter,
						 double targetError)
{
	auto random = std::bind(std::uniform_real_distribution<double>(0, 1),std::ref(generator));
	
	//initialize the walker
	Determinant d;
	bool readDeterminant = false;
	char file [5000];
	sprintf (file, "BestDeterminant.txt");
	
	{
		ifstream ofile(file);
		if (ofile) readDeterminant = true;
	}
	
	if ( !readDeterminant )
	{
		for (int i =0; i<nalpha; i++) {
			int bestorb = 0;
			double maxovlp = 0;
      			for (int j=0; j<norbs; j++) {
				if (abs(HforbsA(i,j)) > maxovlp && !d.getoccA(j)) {
          			maxovlp = abs(HforbsA(i,j));
          			bestorb = j;
       			 	}		
      			}
      			d.setoccA(bestorb, true);
    		}		
    		for (int i =0; i<nbeta; i++) {
      			int bestorb = 0;
      			double maxovlp = 0;
      			for (int j=0; j<norbs; j++) {
        			if (abs(HforbsB(i,j)) > maxovlp && !d.getoccB(j)) {
          			bestorb = j; maxovlp = abs(HforbsB(i,j));
        			}
      			}
      			d.setoccB(bestorb, true);
    		}
  	}
  	else {
    		if (commrank == 0) {
      		std::ifstream ifs(file, std::ios::binary);
      		boost::archive::binary_iarchive load(ifs);
      		load >> d;
    		}
#ifndef SERIAL
    MPI_Bcast(&d.reprA, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d.reprB, DetLen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
	}

        
	Walker walk(d);
  	walk.initUsingWave(w);

	int maxTerms =  (nalpha) * (norbs-nalpha);
	vector<double> ovlpRatio(maxTerms);
	vector<size_t> excitation1( maxTerms), excitation2( maxTerms);
	vector<double> HijElements(maxTerms);
	int nExcitations = 0;

	stddev = 1.e4;
	int iter = 0;
	double M1 = 0., S1 = 0., Eavg = 0.;
	double Eloc = 0.;
	double ham = 0., ovlp = 0.;
	VectorXd localGrad = grad;
	localGrad.setZero();
	double scale = 1.0;

	VectorXd diagonalGrad = VectorXd::Zero(grad.rows());
	VectorXd localdiagonalGrad = VectorXd::Zero(grad.rows());

	double bestOvlp =0.;
	Determinant bestDet=d;

	//new variables for variance
	double Eloc2 = 0.;
	VectorXd gradone = VectorXd::Zero(grad.rows());
	VectorXd gradtwo = VectorXd::Zero(grad.rows());
	VectorXd gradthree = VectorXd::Zero(grad.rows());
	VectorXd gradfour = VectorXd::Zero(grad.rows());

	//find the best determinant at overlap
	if (!readDeterminant) {
		for (int i=0; i<0; i++) {
			w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio, excitation1, excitation2, HijElements, nExcitations, true, true);

      int bestDet = 0;
      double bestOvlp = 0.;
      for (int j=0; j<nExcitations; j++) {
        if (abs(ovlpRatio[j]) > bestOvlp) {
          bestOvlp = abs(ovlpRatio[j]);
          bestDet = j;
        }
      }

      int I = excitation1[bestDet] / 2/ norbs, A = excitation1[bestDet] - 2 * norbs * I;

      if (I % 2 == 0)
        walk.updateA(I / 2, A / 2, w);
      else
        walk.updateB(I / 2, A / 2, w);
      if (excitation2[bestDet] != 0) {
        int J = excitation2[bestDet] / 2 / norbs, B = excitation2[bestDet] - 2 * norbs * J;
        if (J %2 == 1){
          walk.updateB(J / 2, B / 2, w);
        }
        else {
          walk.updateA(J / 2, B / 2, w);
        }

      }
      nExcitations = 0;
    }

    walk.initUsingWave(w, true);
  }

  nExcitations = 0;
  E0 = 0.0;
  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
                       excitation1, excitation2, HijElements, nExcitations, true, true);
  w.OverlapWithGradient(walk, scale, localdiagonalGrad);
  for (int k = 0; k < w.ciExpansion.size(); k++)
  {
    localdiagonalGrad(k + w.getNumJastrowVariables()) += walk.alphaDet[k] * walk.betaDet[k]/ovlp;
  }
  //localGrad = localdiagonalGrad * ham ;
  //we wanto 100000 steps
	int nstore = 1000000/commsize;
	int gradIter = min(nstore, niter);

  std::vector<double> gradError(gradIter*commsize, 0);
  bool reset = true;
  double cumdeltaT = 0., cumdeltaT2 = 0.;
  int reset_len = readDeterminant ? 1 : 10*norbs;
  while (iter < niter && stddev > targetError)
  {
    if (iter == reset_len && reset)
      {
        iter = 0;
        reset = false;
        M1 = 0.;
        S1 = 0.;
        Eloc = 0;
        grad.setZero();
        diagonalGrad.setZero();
        walk.initUsingWave(w, true);
        cumdeltaT = 0.;
        cumdeltaT2 = 0;
	Eloc2 = 0.;
	gradone.setZero();
	gradtwo.setZero();
	gradthree.setZero();
	gradfour.setZero();
      }

    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < nExcitations; i++)
    {
      cumovlpRatio += abs(ovlpRatio[i]);
      //cumovlpRatio += min(1.0, pow(ovlpRatio[i], 2));
      ovlpRatio[i] = cumovlpRatio;
    }

    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random()*cumovlpRatio;
    int nextDet = std::lower_bound(ovlpRatio.begin(), (ovlpRatio.begin()+nExcitations),
                                   nextDetRandom) - ovlpRatio.begin();

    cumdeltaT += deltaT;
    cumdeltaT2 += deltaT * deltaT;

    double Elocold = Eloc;

    double ratio = deltaT/cumdeltaT;
    for (int i=0; i<grad.rows(); i++) {
      gradone(i) += ratio * ( (ham * localGrad(i)) - gradone(i));
      gradtwo(i) += ratio * ( (ham * ham * localdiagonalGrad(i)) - gradtwo(i));
      gradthree(i) += ratio * ( localGrad(i) - gradthree(i));
      gradfour(i) += ratio * ( (ham * localdiagonalGrad(i)) - gradfour(i));
      
      localdiagonalGrad[i] = 0.0;
      localGrad(i) = 0.0;
    }

    Eloc = Eloc + deltaT * (ham - Eloc) / (cumdeltaT);       //running average of energy
    S1 = S1 + (ham - Elocold) * (ham - Eloc);

    Eloc2 = Eloc2 + ratio * ( (ham * ham) - Eloc2);

    if (iter < gradIter)
      gradError[iter + commrank*gradIter] = ham;

    iter++;

    //update walker
    if (true)
    {
      int I = excitation1[nextDet] / 2 / norbs, A = excitation1[nextDet] - 2 * norbs * I;
      int J = excitation2[nextDet] / 2 / norbs, B = excitation2[nextDet] - 2 * norbs * J;


      if (I % 2 == J % 2 && excitation2[nextDet] != 0)
      {
        if (I % 2 == 1) {
          walk.updateB(I / 2, J / 2, A / 2, B / 2, w);
        }
        else {
          walk.updateA(I / 2, J / 2, A / 2, B / 2, w);
        }
      }
      else
      { 
        if (I % 2 == 0)
          walk.updateA(I / 2, A / 2, w);
        else
          walk.updateB(I / 2, A / 2, w);

        if (excitation2[nextDet] != 0)
        {
          if (J % 2 == 1)
          {
            walk.updateB(J / 2, B / 2, w);
          }
          else
          {
            walk.updateA(J / 2, B / 2, w);
          }
       }
      }
    }

    nExcitations = 0;

    w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
                         excitation1, excitation2, HijElements, nExcitations, true, true);
    w.OverlapWithGradient(walk.d, scale, localdiagonalGrad);
    double detovlp=0;
    for (int k = 0; k < w.ciExpansion.size(); k++)
      {
        localdiagonalGrad(k + w.getNumJastrowVariables()) += walk.alphaDet[k] * walk.betaDet[k]/ovlp;
        detovlp += w.ciExpansion[k]*walk.alphaDet[k] * walk.betaDet[k];
      }

    if (w.determinants.size() <= 1 && schd.optimizeOrbs) {
      walk.OverlapWithGradient(w, localdiagonalGrad, detovlp);
    }

  
    if (abs(ovlp) > bestOvlp) {
      bestOvlp = abs(ovlp);
      bestDet = walk.d;
    }
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradone[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradtwo[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradthree[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradfour[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //if (commrank == 0) { std::cout << "I work up to here " << Eloc << std::endl; }
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //if (commrank == 0) { std::cout << "hello" << std::endl; }
  MPI_Allreduce(MPI_IN_PLACE, &Eloc2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  rk = calcTcorr(gradError);

  diagonalGrad /= (commsize);
  gradone /= (commsize);
  gradtwo /= (commsize);
  gradthree /= (commsize);
  gradfour /= (commsize);

  Eloc2 /= (commsize);
  E0 = Eloc / commsize;
  stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);

  var = Eloc2 - (E0 * E0);
  grad = 2.0 * ( gradone - gradtwo + E0 * (gradfour - gradthree));
#ifndef SERIAL
  MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  if (commrank == 0) {
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  }
}

