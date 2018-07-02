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

void comb(int N, int K, vector<vector<int>> &combinations)
{
  std::vector<int> bitmask(K, 1);
  bitmask.resize(N, 0); // N-K trailing 0's

  // print integers and permute bitmask
  int index = 0;
  do
  {
    vector<int> comb;
    for (int i = 0; i < N; ++i) // [0..N-1] integers
    {
      if (bitmask[i] == 1)
        comb.push_back(i);
    }
    combinations.push_back(comb);
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
}

double calcTcorr(vector<double> &v)
{
  vector<double> w(v.size(), 1);
  int n = w.size();
  double norm, rk, f, neff;

  double aver = 0, var = 0;
  for (int i = 0; i < w.size(); i++)
  {
    aver += v[i] * w[i];
    norm += w[i];
  }
  aver = aver / norm;

  neff = 0.0;
  for (int i = 0; i < n; i++)
  {
    neff = neff + w[i] * w[i];
  };
  neff = norm * norm / neff;

  for (int i = 0; i < v.size(); i++)
  {
    var = var + w[i] * (v[i] - aver) * (v[i] - aver);
  };
  var = var / norm;
  var = var * neff / (neff - 1.0);

  double c[w.size()];
  int l = w.size() - 1;
  for (int i = 1; i < l; i++)
  {
    c[i] = 0.0;
    norm = 0.0;
    for (int k = 0; k < n - i; k++)
    {
      c[i] = c[i] + sqrt(w[k] * w[k + i]) * (v[k] - aver) * (v[k + i] - aver);
      norm = norm + sqrt(w[k] * w[k + i]);
    };
    c[i] = c[i] / norm / var;
  };
  rk = 1.0;
  f = 1.0;
  for (int i = 1; i < l; i++)
  {
    if (c[i] < 0.0)
      f = 0.0;
    rk = rk + 2.0 * c[i] * f;
  };

  return rk;
}

void generateAllDeterminants(vector<Determinant>& allDets, int norbs, int nalpha, int nbeta) {
  vector<vector<int>> alphaDets, betaDets;
  comb(norbs, nalpha, alphaDets);
  comb(norbs, nbeta, betaDets);
  
  for (int a = 0; a < alphaDets.size(); a++)
    for (int b = 0; b < betaDets.size(); b++)
    {
      Determinant d;
      for (int i = 0; i < alphaDets[a].size(); i++)
        d.setoccA(alphaDets[a][i], true);
      for (int i = 0; i < betaDets[b].size(); i++)
        d.setoccB(betaDets[b][i], true);
      allDets.push_back(d);
    }

  alphaDets.clear();
  betaDets.clear();
}

//<psi_t| (H-E0) |psi>
void getGradientDeterministic(CPSSlater &w, double &E0, int &nalpha, int &nbeta, int &norbs,
                              oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
                              VectorXd &grad)
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
      w.HamAndOvlpGradient(walk, ovlp, ham, localgrad, I1, I2, I2hb, coreE, ovlpRatio,
                           excitation1, excitation2, HijElements, nExcitations, false, false);

      double tmpovlp = 1.0;
      w.OverlapWithGradient(walk, tmpovlp, localdiagonalGrad);
      double detovlp = walk.getDetOverlap(w);
      for (int k=0; k<w.ciExpansion.size(); k++) {
        localdiagonalGrad(k+w.getNumJastrowVariables()) += walk.alphaDet[k]*walk.betaDet[k]/detovlp;
      }
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
void getGradientHessianDeterministic(CPSSlater &w, double &E0, int &nalpha, int &nbeta, int &norbs,
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
    Walker walk(allDets[i]);
    walk.initUsingWave(w);
    double ovlp = 0, ham = 0;

    {
      E0 = 0.;
      double scale = 1.0;
      localgrad.setZero();
      localdiagonalGrad.setZero();
      w.HamAndOvlpGradient(walk, ovlp, ham, localgrad, I1, I2, I2hb, coreE, ovlpRatio,
                           excitation1, excitation2, HijElements, nExcitations, true, false);

      double tmpovlp = 1.0;
      w.OverlapWithGradient(walk, tmpovlp, localdiagonalGrad);
      double detovlp = walk.getDetOverlap(w);
      for (int k=0; k<w.ciExpansion.size(); k++) {
        localdiagonalGrad(k+w.getNumJastrowVariables()) += walk.alphaDet[k]*walk.betaDet[k]/detovlp;
      }
      //cout << detovlp<<"  "<<localdiagonalGrad[grad.rows()-1]<<"  "<<walk.alphaDet[2]*walk.betaDet[2]<<endl;
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


//<psi|H|psi>/<psi|psi> = <psi|d> <d|H|psi>/<psi|d><d|psi>
double evaluateEDeterministic(CPSSlater &w, int &nalpha, int &nbeta, int &norbs,
                              oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb,
                              double &coreE)
{

  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  VectorXd localGrad;
  bool doGradient = false;
  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;
  vector<double> HijElements;
  int nExcitations = 0;

  double E = 0, ovlp = 0;
  for (int d = commrank; d < allDets.size(); d += commsize)
  {
    double Eloc = 0, ovlploc = 0;
    Walker walk(allDets[d]);
    walk.initUsingWave(w);

    w.HamAndOvlpGradient(walk, ovlploc, Eloc, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
                         excitation1, excitation2, HijElements, nExcitations, doGradient,
                         false);

    E += ovlploc * ovlploc * Eloc;
    ovlp += ovlploc * ovlploc;
  }
  allDets.clear();

  double Ebkp = E, obkp = ovlp;
  int size = 1;
#ifndef SERIAL
  MPI_Allreduce(&Ebkp, &E, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&obkp, &ovlp, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  return E / ovlp;
}

//<psi|H|psi>/<psi|psi> = <psi|d> <d|H|psi>/<psi|d><d|psi>
double evaluateEStochastic(CPSSlater &w, int &nalpha, int &nbeta, int &norbs,
                           oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb,
                           double &coreE, double &stddev,
                           int niter, double targetError)
{

  //initialize the walker
  Determinant d;
  for (int i = 0; i < nalpha; i++)
    d.setoccA(i, true);
  for (int j = 0; j < nbeta; j++)
    d.setoccB(norbs-1-j, true);
  Walker walk(d);
  walk.initUsingWave(w);

  VectorXd localGrad;
  bool doGradient = false;
  int maxTerms =  (nalpha+nbeta) * (nalpha+nbeta) * norbs * norbs;
  vector<double> ovlpRatio(maxTerms);
  vector<size_t> excitation1(maxTerms), excitation2(maxTerms);
  vector<double> HijElements(maxTerms);
  int nExcitations = 0;

  stddev = 1.e4;
  int iter = 0;
  double M1 = 0., S1 = 0., Eavg = 0.;
  double Eloc = 0.;
  double ham = 0., ovlp = 0.;
  double scale = 1.0, Epsi = 0;

  double E0 = 0.0, rk = 1.;
  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE,
                       ovlpRatio, excitation1, excitation2, HijElements, nExcitations,
                       doGradient, false);

  int gradIter = min(niter, 10000);
  std::vector<double> gradError(gradIter, 0);
  bool reset = true;

  while (iter < niter && stddev > targetError)
  {
    if (iter == 100 && reset)
    {
      iter = 0;
      reset = false;
      M1 = 0.;
      S1 = 0.;
      Eloc = 0;
      walk.initUsingWave(w, true);
    }

    Eloc = Eloc + (ham - Eloc) / (iter + 1); //running average of energy

    double Mprev = M1;
    M1 = Mprev + (ham - Mprev) / (iter + 1);
    if (iter != 0)
      S1 = S1 + (ham - Mprev) * (ham - M1);

    if (iter < gradIter)
      gradError[iter] = ham;

    iter++;

    if (iter == gradIter - 1)
    {
      rk = calcTcorr(gradError);
    }

    bool success = walk.makeMove(w);

    if (success)
    {
      w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE,
                           ovlpRatio, excitation1, excitation2, HijElements, nExcitations, 
                           doGradient, false);
      //w.HamAndOvlp(walk, ovlp, ham, I1, I2, I2hb, coreE);
    }
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  E0 = Eloc / commsize;

  stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);
#ifndef SERIAL
  MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  return E0;
}

void getStochasticGradient(CPSSlater &w, double &E0, double &stddev,
                           int &nalpha, int &nbeta, int &norbs,
                           oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
                           VectorXd &grad, double &rk,
                           int niter, double targetError)
{

  //initialize the walker
  Determinant d;
  for (int i = 0; i < nalpha; i++)
    d.setoccA(i, true);
  for (int j = 0; j < nbeta; j++)
    d.setoccB(norbs - 1 - j, true);
  Walker walk(d);
  walk.initUsingWave(w);
  //cout << d <<endl;

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
  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;
  vector<double> HijElements;
  int nExcitations;

  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE,
                       ovlpRatio, excitation1, excitation2, HijElements,
                       nExcitations, true, false);
  w.OverlapWithGradient(walk, scale, localdiagonalGrad);

  int gradIter = min(niter, 10000);
  std::vector<double> gradError(gradIter, 0);
  bool reset = true;

  while (iter < niter && stddev > targetError)
  {
    if (iter == 100 && reset)
    {
      iter = 0;
      reset = false;
      M1 = 0.;
      S1 = 0.;
      Eloc = 0;
      grad.setZero();
      diagonalGrad.setZero();
      walk.initUsingWave(w, true);
    }

    diagonalGrad = diagonalGrad + (localdiagonalGrad - diagonalGrad) / (iter + 1);
    grad = grad + (localGrad - grad) / (iter + 1); //running average of grad
    Eloc = Eloc + (ham - Eloc) / (iter + 1);       //running average of energy

    double Mprev = M1;
    M1 = Mprev + (ham - Mprev) / (iter + 1);
    if (iter != 0)
      S1 = S1 + (ham - Mprev) * (ham - M1);

    if (iter < gradIter)
      gradError[iter] = ham;

    iter++;

    if (iter == gradIter - 1)
    {
      rk = calcTcorr(gradError);
    }

    bool success = walk.makeMove(w);

    if (success)
    {
      localGrad.setZero();
      localdiagonalGrad.setZero();
      w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE,
                           ovlpRatio, excitation1, excitation2, HijElements,
                           nExcitations, true, false);
      w.OverlapWithGradient(walk, scale, localdiagonalGrad);
    }
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  diagonalGrad /= (commsize);
  grad /= (commsize);
  E0 = Eloc / commsize;
  grad = grad - E0 * diagonalGrad;

  stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);
#ifndef SERIAL
  MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
}

void getStochasticGradientContinuousTime(CPSSlater &w, double &E0, double &stddev,
                                         int &nalpha, int &nbeta, int &norbs,
                                         oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
                                         VectorXd &grad, double &rk,
                                         int niter, double targetError)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));
  
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
    for (int i = 0; i < nalpha; i++)
      d.setoccA(i, true);
    for (int j = 0; j < nbeta; j++)
      d.setoccB(j, true);
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
  
  
  int maxTerms =  (nalpha) * (nbeta) * (norbs-nalpha) * (norbs-nbeta);
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
  
  //find the best determinant at overlap
  if (!readDeterminant) {
    for (int i=0; i<50; i++) {
      w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, 
			   I1, I2, I2hb, coreE, ovlpRatio,
			   excitation1, excitation2, HijElements,
			   nExcitations, false);
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
  
  
  E0 = 0.0;
  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
                       excitation1, excitation2, HijElements, nExcitations, false);
  w.OverlapWithGradient(walk, scale, localdiagonalGrad);
  for (int k = 0; k < w.ciExpansion.size(); k++)
  {
    localdiagonalGrad(k + w.getNumJastrowVariables()) += walk.alphaDet[k] * walk.betaDet[k]/ovlp;
  }
  localGrad = localdiagonalGrad * ham ;
  int gradIter = min(niter, 100000);
  std::vector<double> gradError(gradIter, 0);
  bool reset = true;
  double cumdeltaT = 0., cumdeltaT2 = 0.;
  while (iter < niter && stddev > targetError)
  {
    if (iter == 1 && reset)
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
      }
    
    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < nExcitations; i++)
    {
      cumovlpRatio += abs(ovlpRatio[i]);
      //cumovlpRatio += min(1.0, pow(ovlpRatio[i], 2));
      ovlpRatio[i] = cumovlpRatio;

    }

    //double deltaT = -log(random())/(cumovlpRatio);
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random()*cumovlpRatio;
    int nextDet = std::lower_bound(ovlpRatio.begin(), (ovlpRatio.begin()+nExcitations),
                                   nextDetRandom) -
      ovlpRatio.begin();
    
    cumdeltaT += deltaT;
    cumdeltaT2 += deltaT * deltaT;
    
    double Elocold = Eloc;
    
    diagonalGrad = diagonalGrad + deltaT * (localdiagonalGrad - diagonalGrad) / (cumdeltaT);
    grad = grad + deltaT * (localGrad - grad) / (cumdeltaT); //running average of grad
    Eloc = Eloc + deltaT * (ham - Eloc) / (cumdeltaT);       //running average of energy
    
    S1 = S1 + (ham - Elocold) * (ham - Eloc);
    
    if (iter < gradIter)
      gradError[iter] = ham;
    
    iter++;
    
    if (iter == gradIter - 1)
    {
      rk = calcTcorr(gradError);
    }
    
    //update the walker
    if (true)
    {
      //Walker walkbkp = walk;
      
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

    //ovlpRatio.clear();
    //excitation1.clear();
    //excitation2.clear();
    nExcitations = 0;
    
    //localGrad.setZero();
    localdiagonalGrad.setZero();
    w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
			 excitation1, excitation2, HijElements, nExcitations, false);
    w.OverlapWithGradient(walk.d, scale, localdiagonalGrad);
    for (int k = 0; k < w.ciExpansion.size(); k++)
      {
	localdiagonalGrad(k + w.getNumJastrowVariables()) += walk.alphaDet[k] * walk.betaDet[k]/ovlp;
      }
    localGrad = localdiagonalGrad * ham;
    
    if (abs(ovlp) > bestOvlp) {
      bestOvlp = abs(ovlp);
      bestDet = walk.d;
    }
  }
  
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  
  diagonalGrad /= (commsize);
  grad /= (commsize);
  E0 = Eloc / commsize;
  grad = grad - E0 * diagonalGrad;
  
  stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);
#ifndef SERIAL
  MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  if (commrank == 0) {
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  }
}


void getStochasticGradientHessianContinuousTime(CPSSlater &w, double &E0, double &stddev,
                                         int &nalpha, int &nbeta, int &norbs,
                                         oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE,
                                         VectorXd &grad, MatrixXd& Hessian, MatrixXd& Smatrix, double &rk,
                                         int niter, double targetError)
{
  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  //initialize the walker
  Determinant d;
  for (int i = 0; i < nalpha; i++)
    d.setoccA(i, true);
  for (int j = 0; j < nbeta; j++)
    d.setoccB(j, true);
  Walker walk(d);
  walk.initUsingWave(w);
  //cout << d <<endl;

  int maxTerms =  (nalpha+nbeta) * (nalpha+nbeta) * norbs * norbs;
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

  E0 = 0.0;
  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
                       excitation1, excitation2, HijElements, nExcitations, true, true);
  w.OverlapWithGradient(walk, scale, localdiagonalGrad);
  double detovlp = walk.getDetOverlap(w);
  for (int k = 0; k < w.ciExpansion.size(); k++)
  {
    localdiagonalGrad(k + w.getNumJastrowVariables()) += walk.alphaDet[k] * walk.betaDet[k]/detovlp;
  }
  grad = localdiagonalGrad * ham ;

  //Hessian = localGrad*localdiagonalGrad.transpose();
  //Smatrix = localdiagonalGrad*localdiagonalGrad.transpose();

  int gradIter = min(niter, 100000);
  std::vector<double> gradError(gradIter, 0);
  bool reset = true;
  double cumdeltaT = 0., cumdeltaT2 = 0.;
  while (iter < niter && stddev > targetError)
  {
    if (iter == 100 && reset)
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
      Hessian.setZero(); Smatrix.setZero();
    }
    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < nExcitations; i++)
    {
      cumovlpRatio += ovlpRatio[i];
      //cumovlpRatio += min(1.0, pow(ovlpRatio[i], 2));
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

    if (!reset) {
      Hessian.block(1, 1, grad.rows(), grad.rows()) += deltaT * (localGrad * localdiagonalGrad.transpose() - Hessian.block(1, 1, grad.rows(), grad.rows())) / cumdeltaT;
      Smatrix.block(1, 1, grad.rows(), grad.rows()) += deltaT * (localdiagonalGrad * localdiagonalGrad.transpose() - Smatrix.block(1, 1, grad.rows(), grad.rows())) / cumdeltaT;
      Hessian.block(0, 1, 1, grad.rows()) += deltaT * (localGrad.transpose() - Hessian.block(0, 1, 1, grad.rows())) / cumdeltaT;
      Hessian.block(1, 0, grad.rows(), 1) += deltaT * (localGrad - Hessian.block(1, 0, grad.rows(), 1)) / cumdeltaT;
      Smatrix.block(0, 1, 1, grad.rows()) += deltaT * (localdiagonalGrad.transpose() - Smatrix.block(0, 1, 1, grad.rows())) / cumdeltaT;
      Smatrix.block(1, 0, grad.rows(), 1) += deltaT * (localdiagonalGrad - Smatrix.block(1, 0, grad.rows(), 1)) / cumdeltaT;
    }

    S1 = S1 + (ham - Elocold) * (ham - Eloc);

    if (iter < gradIter)
      gradError[iter] = ham;

    iter++;

    if (iter == gradIter - 1)
    {
      rk = calcTcorr(gradError);
    }

    //update the walker
    if (true)
    {
      int I = excitation1[nextDet] / 2 / norbs, A = excitation1[nextDet] - 2 * norbs * I;
      if (I % 2 == 0)
        walk.updateA(I / 2, A / 2, w);
      else
        walk.updateB(I / 2, A / 2, w);

      if (excitation2[nextDet] != 0) {
        int J = excitation2[nextDet] / 2 / norbs, B = excitation2[nextDet] - 2 * norbs * J;
        if (J %2 == 1){
          walk.updateB(J / 2, B / 2, w);
        }
        else {
          walk.updateA(J / 2, B / 2, w);          
        }
      }
      //ovlpRatio.clear();
      //excitation1.clear();
      //excitation2.clear();
      nExcitations = 0;

      localGrad.setZero();
      localdiagonalGrad.setZero();
      w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
                           excitation1, excitation2, HijElements, nExcitations, true, true);
      w.OverlapWithGradient(walk.d, scale, localdiagonalGrad);
      double detovlp = walk.getDetOverlap(w);
      for (int k = 0; k < w.ciExpansion.size(); k++)
      {
        localdiagonalGrad(k + w.getNumJastrowVariables()) += walk.alphaDet[k] * walk.betaDet[k]/detovlp;
      }
      //localGrad = localdiagonalGrad * ham;
    }
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Hessian(0,0)), Hessian.rows()*Hessian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Smatrix.rows()*Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

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

}
