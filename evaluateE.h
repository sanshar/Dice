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

//evaluate energy and gradient using stochastic or deterministic algorithm
double evaluateEDeterministic(CPSSlater &w, int &nalpha, int &nbeta, int &norbs,
							  oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb,
							  double &coreE);

double evaluateEStochastic(CPSSlater &w, int &nalpha, int &nbeta, int &norbs,
						   oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb,
						   double &coreE, double &stddev,
						   int niter = 10000, double targetError = 1.e-3);

void getGradientDeterministic(CPSSlater &w, double &E0, int &alpha, int &nbeta, int &norbs,
							  oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
							  Eigen::VectorXd &grad);

void getGradientHessianDeterministic(CPSSlater &w, double &E0, int &alpha, int &nbeta, int &norbs,
									 oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
									 Eigen::VectorXd &grad, Eigen::MatrixXd &Hessian,
									 Eigen::MatrixXd &Smatrix);

void getStochasticGradient(CPSSlater &w, double &E0, double &stddev,
						   int &nalpha, int &nbeta, int &norbs,
						   oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
						   Eigen::VectorXd &grad, double &rk,
						   int niter, double targetError);


template <class Wfn, class Walker>
void getStochasticGradientContinuousTime(Wfn &w, Walker &walk, double &E0, double &stddev,
										 Eigen::VectorXd &grad, double &rk,
										 int niter, double targetError)
{
	int norbs = Determinant::norbs;
	int nalpha = Determinant::nalpha;
	int nbeta = Determinant::nbeta;

	auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
							std::ref(generator));


	//int maxTerms =  3*(nalpha) * (nbeta) * (norbs-nalpha) * (norbs-nbeta);
	int maxTerms = (nalpha) * (norbs - nalpha); //pick a small number that will be incremented later
	vector<double> ovlpRatio(maxTerms);
	vector<size_t> excitation1(maxTerms), excitation2(maxTerms);
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

	double bestOvlp = 0.;
	Determinant bestDet = walk.d;

	nExcitations = 0;
	E0 = 0.0;
	w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
						 excitation1, excitation2, HijElements, nExcitations, false);
	w.OverlapWithGradient(walk, scale, localdiagonalGrad);

	int nstore = 1000000 / commsize;
	int gradIter = min(nstore, niter);

	std::vector<double> gradError(gradIter * commsize, 0);
	bool reset = true;
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
		double nextDetRandom = random() * cumovlpRatio;
		int nextDet = std::lower_bound(ovlpRatio.begin(), (ovlpRatio.begin() + nExcitations),
									   nextDetRandom) -
					  ovlpRatio.begin();

		cumdeltaT += deltaT;
		cumdeltaT2 += deltaT * deltaT;

		double Elocold = Eloc;

		double ratio = deltaT / cumdeltaT;
		for (int i = 0; i < grad.rows(); i++)
		{
			diagonalGrad[i] += ratio * (localdiagonalGrad[i] - diagonalGrad[i]);
			//localGrad[i] = ham*localdiagonalGrad[i];
			grad[i] += ratio * (ham * localdiagonalGrad[i] - grad[i]);
			localdiagonalGrad[i] = 0.0;
		}

		//cout <<nextDet<<"  "<< excitation1[nextDet]<<"  "<<excitation2[nextDet]<<endl;
		//cout << walk.d <<"  "<<Eloc<<"  "<<grad.norm()<<endl;

		Eloc = Eloc + deltaT * (ham - Eloc) / (cumdeltaT); //running average of energy

		S1 = S1 + (ham - Elocold) * (ham - Eloc);

		if (iter < gradIter)
			gradError[iter + commrank * gradIter] = ham;

		iter++;

		walk.updateWalker(w, excitation1[nextDet], excitation2[nextDet]);

		nExcitations = 0;
		w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
							 excitation1, excitation2, HijElements, nExcitations, false);
		w.OverlapWithGradient(walk.d, scale, localdiagonalGrad);

		if (abs(ovlp) > bestOvlp)
		{
			bestOvlp = abs(ovlp);
			bestDet = walk.d;
		}
	}

#ifndef SERIAL
	MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

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

void getStochasticGradientHessianContinuousTime(CPSSlater &w, double &E0, double &stddev,
												int &nalpha, int &nbeta, int &norbs,
												oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
												Eigen::VectorXd &grad, Eigen::MatrixXd &Hessian, Eigen::MatrixXd &Smatrix, double &rk,
												int niter, double targetError);

#endif
