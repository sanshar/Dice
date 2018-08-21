/*
  Developed by Sandeep Sharma
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
#ifndef EvalE_HEADER_H
#define EvalE_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "Determinants.h"
#include "workingArray.h"
#include <iostream>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace Eigen;
using namespace std;

class Walker;
class Wfn;
class CPSSlater;
class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class MoDeterminant;

//generate all the alpha or beta strings
void comb(int N, int K, std::vector<std::vector<int>> &combinations);

//calculate reblocking analysis to find correlation length
double calcTcorr(std::vector<double> &v);

void generateAllDeterminants(vector<Determinant> &allDets, int norbs, int nalpha, int nbeta);

template<typename Wfn, typename Walker> void getGradientDeterministic(Wfn &w, Walker& walk, double &E0,
                              VectorXd &grad)
{
  int norbs = Determinant::norbs;
	int nalpha = Determinant::nalpha;
	int nbeta = Determinant::nbeta;

  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  workingArray work;

  double Overlap = 0, Energy = 0;
  grad.setZero();
  VectorXd diagonalGrad = VectorXd::Zero(grad.rows());
  VectorXd localdiagonalGrad = VectorXd::Zero(grad.rows());
  VectorXd localgrad = VectorXd::Zero(grad.rows());

  for (int i = commrank; i < allDets.size(); i += commsize)
  {
    w.initWalker(walk, allDets[i]);
    double ovlp = 0, ham = 0;
    {
      E0 = 0.;
      double scale = 1.0;
      localgrad.setZero();
      localdiagonalGrad.setZero();
      w.HamAndOvlp(walk, ovlp, ham, work, false);

      double tmpovlp = 1.0;
      w.OverlapWithGradient(walk, ovlp, localdiagonalGrad);
    }
    
    //grad += localgrad * ovlp * ovlp;
    grad += localdiagonalGrad * ham * ovlp * ovlp;
    diagonalGrad += localdiagonalGrad * ovlp * ovlp;
    Overlap += ovlp * ovlp;
    Energy += ham * ovlp * ovlp;
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  E0 = Energy / Overlap;
  grad = (grad - E0 * diagonalGrad) / Overlap;
}

//<psi_t| (H-E0) |psi>
template<typename Wfn, typename Walker> void getGradientHessianDeterministic(Wfn &w, Walker& walk, double &E0, int &nalpha, int &nbeta, int &norbs,
                                    oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
                                    VectorXd &grad, MatrixXd& Hessian, MatrixXd &Smatrix)
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
  double temp = 0.;

  for (int i = commrank; i < allDets.size(); i += commsize)
  {
    w.initWalker(walk, allDets[i]);
    double ovlp = 0, ham = 0;

    {
      E0 = 0.;
      double scale = 1.0;
      localgrad.setZero();
      localdiagonalGrad.setZero();
      w.HamAndOvlpGradient(walk, ovlp, ham, localgrad, I1, I2, I2hb, coreE, ovlpRatio,
                           excitation1, excitation2, HijElements, nExcitations, true, false);

      double tmpovlp = 1.0;
      w.OverlapWithGradient(walk, ovlp, localdiagonalGrad);
    }
    //grad += localgrad * ovlp * ovlp;
    grad += localdiagonalGrad * ham * ovlp * ovlp;
    diagonalGrad += localdiagonalGrad * ovlp * ovlp;
    Overlap += ovlp * ovlp;
    Energy += ham * ovlp * ovlp;

    Hessian.block(1, 1, grad.rows(), grad.rows()) += localgrad * localdiagonalGrad.transpose() * ovlp * ovlp;
    Smatrix.block(1, 1, grad.rows(), grad.rows()) += localdiagonalGrad * localdiagonalGrad.transpose() * ovlp * ovlp;
    Hessian.block(0, 1, 1, grad.rows()) += ovlp * ovlp * localgrad.transpose();
    Hessian.block(1, 0, grad.rows(), 1) += ovlp * ovlp * localgrad;
    Smatrix.block(0, 1, 1, grad.rows()) += ovlp * ovlp * localdiagonalGrad.transpose();
    Smatrix.block(1, 0, grad.rows(), 1) += ovlp * ovlp * localdiagonalGrad;
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Hessian(0,0)), Hessian.rows()*Hessian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Hessian.rows()*Hessian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  E0 = Energy / Overlap;
  grad = (grad - E0 * diagonalGrad) / Overlap;
  Hessian = Hessian/Overlap;
  Smatrix = Smatrix/Overlap;
  Smatrix(0,0) = 1.0;
  Hessian(0,0) = E0;

}


template<typename Wfn, typename Walker> 
  void getStochasticEnergyContinuousTime(Wfn &w, Walker &walk, double &E0, double &stddev,
					 double &rk, int niter, double targetError)
{
	int norbs = Determinant::norbs;
	int nalpha = Determinant::nalpha;
	int nbeta = Determinant::nbeta;

	auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
							std::ref(generator));


	workingArray work;

	stddev = 1.e4;
	int iter = 0;
	double M1 = 0., S1 = 0., Eavg = 0.;
	double Eloc = 0.;
	double ham = 0., ovlp = 0.;
	double scale = 1.0;

	double bestOvlp = 0.;
	Determinant bestDet = walk.getDet();

	E0 = 0.0;
	w.HamAndOvlp(walk, ovlp, ham, work);


	int nstore = 1000000 / commsize;
	int gradIter = min(nstore, niter);

	std::vector<double> gradError(gradIter * commsize, 0);
	double cumdeltaT = 0., cumdeltaT2 = 0.;

	while (iter < niter && stddev > targetError)
	{

		double cumovlpRatio = 0;
		//when using uniform probability 1./numConnection * max(1, pi/pj)
		for (int i = 0; i < work.nExcitations; i++)
		{
			cumovlpRatio += abs(work.ovlpRatio[i]);
			work.ovlpRatio[i] = cumovlpRatio;
		}

		//double deltaT = -log(random())/(cumovlpRatio);
		double deltaT = 1.0 / (cumovlpRatio);
		double nextDetRandom = random() * cumovlpRatio;
		int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
									   nextDetRandom) -
					  work.ovlpRatio.begin();

		cumdeltaT += deltaT;
		cumdeltaT2 += deltaT * deltaT;

		double Elocold = Eloc;

		double ratio = deltaT / cumdeltaT;
		Eloc = Eloc + deltaT * (ham - Eloc) / (cumdeltaT); //running average of energy

		S1 = S1 + (ham - Elocold) * (ham - Eloc);

		if (iter < gradIter)
			gradError[iter + commrank * gradIter] = ham;

		iter++;

		walk.updateWalker(w, work.excitation1[nextDet], work.excitation2[nextDet]);

		w.HamAndOvlp(walk, ovlp, ham, work);
	}

#ifndef SERIAL
	MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

	//if (commrank == 0)
	rk = calcTcorr(gradError);

	E0 = Eloc / commsize;

	stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);
#ifndef SERIAL
	MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
}



template<typename Wfn, typename Walker> void getStochasticGradientContinuousTime(Wfn &w, Walker &walk, double &E0, double &stddev,
										 Eigen::VectorXd &grad, double &rk,
										 int niter, double targetError)
{
	int norbs = Determinant::norbs;
	int nalpha = Determinant::nalpha;
	int nbeta = Determinant::nbeta;

	auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
							std::ref(generator));


	workingArray work;

	stddev = 1.e4;
	int iter = 0;
	double M1 = 0., S1 = 0., Eavg = 0.;
	double Eloc = 0.;
	double ham = 0., ovlp = 0.;
  grad.setZero();
	VectorXd localGrad = grad;
	double scale = 1.0;

	VectorXd diagonalGrad = VectorXd::Zero(grad.rows());
	VectorXd localdiagonalGrad = VectorXd::Zero(grad.rows());

	double bestOvlp = 0.;
	Determinant bestDet = walk.getDet();

	E0 = 0.0;
	w.HamAndOvlp(walk, ovlp, ham, work);
	w.OverlapWithGradient(walk, ovlp, localdiagonalGrad);


	int nstore = 1000000 / commsize;
	int gradIter = min(nstore, niter);

	std::vector<double> gradError(gradIter * commsize, 0);
	double cumdeltaT = 0., cumdeltaT2 = 0.;

	while (iter < niter && stddev > targetError)
	{

		double cumovlpRatio = 0;
		//when using uniform probability 1./numConnection * max(1, pi/pj)
		for (int i = 0; i < work.nExcitations; i++)
		{
			cumovlpRatio += abs(work.ovlpRatio[i]);
			work.ovlpRatio[i] = cumovlpRatio;
		}

		//double deltaT = -log(random())/(cumovlpRatio);
		double deltaT = 1.0 / (cumovlpRatio);
		double nextDetRandom = random() * cumovlpRatio;
		int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
									   nextDetRandom) -
					  work.ovlpRatio.begin();

		cumdeltaT += deltaT;
		cumdeltaT2 += deltaT * deltaT;

		double Elocold = Eloc;

		//exit(0);
		//cout << walk.d<<"  "<<ham<<"  "<<Eloc<<endl;

		double ratio = deltaT / cumdeltaT;
		for (int i = 0; i < grad.rows(); i++)
		{
			diagonalGrad[i] += ratio * (localdiagonalGrad[i] - diagonalGrad[i]);
			grad[i] += ratio * (ham * localdiagonalGrad[i] - grad[i]);
			localdiagonalGrad[i] = 0.0;
		}

		Eloc = Eloc + deltaT * (ham - Eloc) / (cumdeltaT); //running average of energy

		S1 = S1 + (ham - Elocold) * (ham - Eloc);

		if (iter < gradIter)
			gradError[iter + commrank * gradIter] = ham;

		iter++;

		walk.updateWalker(w, work.excitation1[nextDet], work.excitation2[nextDet]);

		w.HamAndOvlp(walk, ovlp, ham, work);
		w.OverlapWithGradient(walk, ovlp, localdiagonalGrad);

		if (abs(ovlp) > bestOvlp)
		{
			bestOvlp = abs(ovlp);
			bestDet = walk.getDet();
		}
	}

#ifndef SERIAL
	MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	//exit(0);
	//if (commrank == 0)
	rk = calcTcorr(gradError);

	diagonalGrad /= (commsize);
	grad /= (commsize);
	E0 = Eloc / commsize;
	grad = grad - E0 * diagonalGrad;

	stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);
#ifndef SERIAL
	MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

	if (commrank == 0)
	{
		char file[5000];
		sprintf(file, "BestDeterminant.txt");
		std::ofstream ofs(file, std::ios::binary);
		boost::archive::binary_oarchive save(ofs);
		save << bestDet;
	}
}

template<typename Wfn, typename Walker> void getStochasticGradientHessianContinuousTime(Wfn &w, Walker& walk, double &E0, double &stddev,
                                         oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
                                         VectorXd &grad, MatrixXd& Hessian, MatrixXd& Smatrix, double &rk,
                                         int niter, double targetError)
{
	int norbs = Determinant::norbs;
	int nalpha = Determinant::nalpha;
	int nbeta = Determinant::nbeta;

	auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
							std::ref(generator));

  int maxTerms =  1000000;
  vector<double> ovlpRatio(maxTerms);
  vector<size_t> excitation1( maxTerms), excitation2( maxTerms);
  vector<double> HijElements(maxTerms);
  int nExcitations = 0;

  stddev = 1.e4;
  int iter = 0;
  double M1 = 0., S1 = 0., Eavg = 0.;
  double Eloc = 0.;
  double ham = 0., ovlp = 0.;
  grad.setZero();
  Hessian.setZero(); Smatrix.setZero();
  VectorXd hamiltonianRatio = grad;
  double scale = 1.0;

  VectorXd diagonalGrad = VectorXd::Zero(grad.rows());
  VectorXd localdiagonalGrad = VectorXd::Zero(grad.rows());

  double bestOvlp =0.;
  Determinant bestDet=walk.getDet();

	nExcitations = 0;
  E0 = 0.0;
  w.HamAndOvlpGradient(walk, ovlp, ham, hamiltonianRatio, I1, I2, I2hb, coreE, ovlpRatio,
                       excitation1, excitation2, HijElements, nExcitations, true, true);
  w.OverlapWithGradient(walk, scale, localdiagonalGrad);

  int nstore = 1000000/commsize;
  int gradIter = min(nstore, niter);

  std::vector<double> gradError(gradIter*commsize, 0);
  double cumdeltaT = 0., cumdeltaT2 = 0.;
  while (iter < niter && stddev > targetError)
  {

    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < nExcitations; i++)
    {
      cumovlpRatio += abs(ovlpRatio[i]);
      ovlpRatio[i] = cumovlpRatio;
    }

    //double deltaT = -log(random())/(cumovlpRatio);
    double deltaT = 1.0 / (cumovlpRatio);
    int nextDet = std::lower_bound(ovlpRatio.begin(), (ovlpRatio.begin()+nExcitations),
                                   random() * cumovlpRatio) -
                  ovlpRatio.begin();

    cumdeltaT += deltaT;
    cumdeltaT2 += deltaT * deltaT;

    double Elocold = Eloc;

    diagonalGrad = diagonalGrad + deltaT * (localdiagonalGrad - diagonalGrad) / (cumdeltaT);
    grad = grad + deltaT * (localdiagonalGrad * ham - grad) / (cumdeltaT); //running average of grad
    Eloc = Eloc + deltaT * (ham - Eloc) / (cumdeltaT);       //running average of energy

    Hessian.block(1, 1, grad.rows(), grad.rows()) += deltaT * (hamiltonianRatio * localdiagonalGrad.transpose() - Hessian.block(1, 1, grad.rows(), grad.rows())) / cumdeltaT;
    Smatrix.block(1, 1, grad.rows(), grad.rows()) += deltaT * (localdiagonalGrad * localdiagonalGrad.transpose() - Smatrix.block(1, 1, grad.rows(), grad.rows())) / cumdeltaT;
    Hessian.block(0, 1, 1, grad.rows()) += deltaT * (hamiltonianRatio.transpose() - Hessian.block(0, 1, 1, grad.rows())) / cumdeltaT;
    Hessian.block(1, 0, grad.rows(), 1) += deltaT * (hamiltonianRatio - Hessian.block(1, 0, grad.rows(), 1)) / cumdeltaT;
    Smatrix.block(0, 1, 1, grad.rows()) += deltaT * (localdiagonalGrad.transpose() - Smatrix.block(0, 1, 1, grad.rows())) / cumdeltaT;
    Smatrix.block(1, 0, grad.rows(), 1) += deltaT * (localdiagonalGrad - Smatrix.block(1, 0, grad.rows(), 1)) / cumdeltaT;

    S1 = S1 + (ham - Elocold) * (ham - Eloc);

    if (iter < gradIter)
      gradError[iter + commrank*gradIter] = ham;

    iter++;

		walk.updateWalker(w, excitation1[nextDet], excitation2[nextDet]);

    nExcitations = 0;
    
    hamiltonianRatio.setZero();
    localdiagonalGrad.setZero();
    w.HamAndOvlpGradient(walk, ovlp, ham, hamiltonianRatio, I1, I2, I2hb, coreE, ovlpRatio,
			 excitation1, excitation2, HijElements, nExcitations, true, true);
    w.OverlapWithGradient(walk, ovlp, localdiagonalGrad);

    if (abs(ovlp) > bestOvlp) {
      bestOvlp = abs(ovlp);
      bestDet = walk.getDet();
    }

  }
  
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Hessian(0,0)), Hessian.rows()*Hessian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Smatrix.rows()*Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  rk = calcTcorr(gradError);

  diagonalGrad /= (commsize);
  grad /= (commsize);
  E0 = Eloc / commsize;
  grad = grad - E0 * diagonalGrad;
  Hessian = Hessian/(commsize);
  Smatrix = Smatrix/(commsize);

  stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);
#ifndef SERIAL
  MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  Smatrix(0,0) = 1.0;
  Hessian(0,0) = E0;

  if (commrank == 0) {
		char file[5000];
		sprintf(file, "BestDeterminant.txt");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  }

}

template <typename Wfn, typename Walker>
class getGradientWrapper
{
public:
  Wfn &w;
  Walker &walk;
  int stochasticIter;
  getGradientWrapper(Wfn &pw, Walker &pwalk, int niter) : w(pw), walk(pwalk)
  {
    stochasticIter = niter;
  };

  void getGradient(VectorXd &vars, VectorXd &grad, double &E0, double &stddev, double &rt, bool deterministic)
  {
    if (!deterministic)
    {
      w.updateVariables(vars);
      w.initWalker(walk);
      getStochasticGradientContinuousTime(w, walk, E0, stddev, grad, rt, stochasticIter, 0.5e-3);
    }
    else
    {
      w.updateVariables(vars);
      w.initWalker(walk);
      stddev = 0.0;
      rt = 1.0;
      getGradientDeterministic(w, walk, E0, grad);
    }
    w.writeWave();
  };

};
#endif

/*void getStochasticGradient(CPSSlater &w, double &E0, double &stddev,
						   int &nalpha, int &nbeta, int &norbs,
						   oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
						   Eigen::VectorXd &grad, double &rk,
						   int niter, double targetError);


*/
