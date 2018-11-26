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
#include "statistics.h"
#include "sr.h"
#include "global.h"
#include <iostream>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <algorithm>

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace Eigen;
using namespace std;
using namespace boost;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;

//generate all the alpha or beta strings
void comb(int N, int K, std::vector<std::vector<int>> &combinations);

//calculate reblocking analysis to find correlation length
double calcTcorr(std::vector<double> &v);

void generateAllDeterminants(vector<Determinant> &allDets, int norbs, int nalpha, int nbeta);

template<typename Wfn, typename Walker>
void getEnergyDeterministic(Wfn &w, Walker& walk, double &E0)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  workingArray work;

  double Overlap = 0, Energy = 0;


  for (int i = commrank; i < allDets.size(); i += commsize)
  {
    w.initWalker(walk, allDets[i]);
    double ovlp = 0, ham = 0;
    {
      E0 = 0.;
      double scale = 1.0;


      w.HamAndOvlp(walk, ovlp, ham, work, false);
    }
    
    //grad += localgrad * ovlp * ovlp;
    Overlap += ovlp * ovlp;
    Energy += ham * ovlp * ovlp;
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  E0 = Energy / Overlap;
}

template<typename Wfn, typename Walker>
void getGradientDeterministic(Wfn &w, Walker& walk, double &E0, VectorXd &grad)
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
      //cout << walk << endl;

      w.HamAndOvlp(walk, ovlp, ham, work, false);
      //cout <<"ham  " << ham << " ovlp  " << ovlp << endl << endl;
      double tmpovlp = 1.0;
      w.OverlapWithGradient(walk, ovlp, localdiagonalGrad);
      //cout << "grad\n" << localdiagonalGrad << endl << endl;
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
  //cout << "totalGrad\n" << grad << endl << endl;
}

template<typename Wfn, typename Walker>
void getGradientHessianDeterministic(Wfn &w, Walker& walk, double &E0, int &nalpha, int &nbeta, int &norbs, oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE, VectorXd &grad, MatrixXd& Hessian, MatrixXd &Smatrix)
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
void getLanczosCoeffsDeterministic(Wfn &w, Walker &walk, double &alpha, Eigen::VectorXd &lanczosCoeffs)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  workingArray work, moreWork;

  double overlapTot = 0.; 
  Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(4);
  //w.printVariables();

  for (int i = commrank; i < allDets.size(); i += commsize)
  {
    w.initWalker(walk, allDets[i]);
    Eigen::VectorXd coeffsSample = Eigen::VectorXd::Zero(4);
    double overlapSample = 0.;
    //cout << walk;
    w.HamAndOvlpLanczos(walk, coeffsSample, overlapSample, work, moreWork, alpha);
    //cout << "ham  " << ham[0] << "  " << ham[1] << "  " << ham[2] << endl;
    //cout << "ovlp  " << ovlp[0] << "  " << ovlp[1] << "  " << ovlp[2] << endl << endl;
    
    //grad += localgrad * ovlp * ovlp;
    overlapTot += overlapSample * overlapSample;
    coeffs += (overlapSample * overlapSample) * coeffsSample;
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(overlapTot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, coeffs.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  lanczosCoeffs = coeffs / overlapTot;
}

template<typename Wfn, typename Walker>
void getGradientMetricDeterministic(Wfn &w, Walker& walk, double &E0, VectorXd &grad, VectorXd &H, DirectMetric &S)
{
  //cout << "1" << endl; 
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  vector<Determinant> allDets;
  generateAllDeterminants(allDets, norbs, nalpha, nbeta);

  workingArray work;

  double Overlap = 0.0, Energy = 0.0;
  grad.setZero();
  int numVars = grad.rows();
  VectorXd grad_ratio_bar = VectorXd::Zero(numVars);
  VectorXd grad_ratio = VectorXd::Zero(numVars);
  S.T.setZero(allDets.size());
  S.Vectors.setZero(numVars + 1, allDets.size());
  H.setZero(numVars + 1);

  //cout << "2" << endl; 
/*
  char sfile[50];
  sprintf(sfile, "./Metric/sVectors%i.bin", commrank);
  ofstream s(sfile, ios::binary);
  char tfile[50];
  sprintf(tfile, "./T/%i.bin", commrank);
  ofstream t(tfile, ios::binary); 
*/

  for (int i = commrank; i < allDets.size(); i += commsize)
  {
    w.initWalker(walk, allDets[i]);
    double ovlp = 0.0, Eloc = 0.0;

    {
      E0 = 0.;
      double scale = 1.0;
      grad_ratio.setZero();
      w.HamAndOvlp(walk, ovlp, Eloc, work, false);
      w.OverlapWithGradient(walk, ovlp, grad_ratio);
    }
    //grad += localgrad * ovlp * ovlp;
    grad += grad_ratio * Eloc * ovlp * ovlp;
    grad_ratio_bar += grad_ratio * ovlp * ovlp;
    Overlap += ovlp * ovlp;
    Energy += Eloc * ovlp * ovlp;

    VectorXd appended(numVars + 1);
    appended << 1.0, grad_ratio;
/*
    t << (ovlp * ovlp);
    s << appended;
*/
    S.Vectors.col(i) = (appended);
    S.T(i) = (ovlp * ovlp);
/*
    appended << 1.0, localdiagonalGrad;
    S.Vectors.col(i) = appended;
    S.T.push_back(ovlp * ovlp);
    Smatrix.block(1, 1, grad.rows(), grad.rows()) += localdiagonalGrad * localdiagonalGrad.transpose() * ovlp * ovlp;
    Smatrix.block(0, 1, 1, grad.rows()) += ovlp * ovlp * localdiagonalGrad.transpose();
    Smatrix.block(1, 0, grad.rows(), 1) += ovlp * ovlp * localdiagonalGrad;
    */
  }
  
  //cout << "3" << endl; 
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad_ratio_bar[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, (S.T.data()), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, (S.Vectors.data()), S.Vectors.rows() * S.Vectors.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  E0 = Energy / Overlap;
  grad = (grad - E0 * grad_ratio_bar) / Overlap;
  //Smatrix = Smatrix/Overlap;
  //Smatrix(0,0) = 1.0;
  VectorXd appended(numVars);
  appended = grad_ratio_bar - schd.stepsize * grad;
  H << 1.0, appended;
  //cout << "4" << endl; 
/*
  s.close();
  t.close();
*/
}

template<typename Wfn, typename Walker> 
void getStochasticEnergyContinuousTime(Wfn &w, Walker &walk, double &E0, double &stddev, double &rk, int niter, double targetError)
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
  std::vector<double> tauError(gradIter * commsize, 0);
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

    S1 = S1 + deltaT * (ham - Elocold) * (ham - Eloc);

    if (iter < gradIter)
    {
      gradError[iter + commrank * gradIter] = ham;
      tauError[iter + commrank * gradIter] = deltaT;
    }

    iter++;

    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    w.HamAndOvlp(walk, ovlp, ham, work);
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(tauError[0]), tauError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  //if (commrank == 0)
  vector<double> r_x, b_size;
  blocking(b_size, r_x, gradError, tauError);
  rk = corr_time(gradError.size(), b_size, r_x);

  E0 = Eloc / commsize;
  S1 /= cumdeltaT;

  double n_eff = commsize * (cumdeltaT * cumdeltaT) / cumdeltaT2;
  stddev =  sqrt(rk * S1 / n_eff);
  //stddev = sqrt(S1 * rk / (niter - 1) / niter / commsize);
#ifndef SERIAL
  MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
}

template<typename Wfn, typename Walker> 
void getLanczosCoeffsContinuousTime(Wfn &w, Walker &walk, double &alpha, Eigen::VectorXd &lanczosCoeffs, Eigen::VectorXd &stddev, Eigen::VectorXd &rk, int niter, double targetError)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                          std::ref(generator));

  int iter = 0;
  Eigen::VectorXd S1 = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd coeffsSample = Eigen::VectorXd::Zero(4);
  double ovlpSample = 0.;

  double bestOvlp = 0.;
  Determinant bestDet = walk.getDet();

  workingArray work, moreWork;
  w.HamAndOvlpLanczos(walk, coeffsSample, ovlpSample, work, moreWork, alpha);

  int nstore = 1000000 / commsize;
  int gradIter = min(nstore, niter);

  std::vector<std::vector<double>> gradError;
  gradError.resize(4);
  //std::vector<double> gradError(gradIter * commsize, 0.);
  vector<double> tauError(gradIter * commsize, 0.);
  for (int i = 0; i < 4; i++)
    gradError[i] = std::vector<double>(gradIter * commsize, 0.);
  double cumdeltaT = 0.;
  double cumdeltaT2 = 0.;
  
  int transIter = 0, nTransIter = 1000;

  while (transIter < nTransIter) {
    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < work.nExcitations; i++) {
      cumovlpRatio += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumovlpRatio;
    }

    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
                                   nextDetRandom) - work.ovlpRatio.begin();

    transIter++;
    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    w.HamAndOvlpLanczos(walk, coeffsSample, ovlpSample, work, moreWork, alpha);
  }

  while (iter < niter) {
    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < work.nExcitations; i++) {
      cumovlpRatio += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumovlpRatio;
    }

    //double deltaT = -log(random())/(cumovlpRatio);
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
                                   nextDetRandom) - work.ovlpRatio.begin();

    cumdeltaT += deltaT;
    cumdeltaT2 += deltaT * deltaT;
    double ratio = deltaT / cumdeltaT;
    Eigen::VectorXd coeffsOld = coeffs;
    coeffs += ratio * (coeffsSample - coeffs);
    S1 += deltaT * (coeffsSample - coeffsOld).cwiseProduct(coeffsSample - coeffs);

    if (iter < gradIter) {
      tauError[iter + commrank * gradIter] = deltaT;
      for (int i = 0; i < 4; i++) 
        gradError[i][iter + commrank * gradIter] = coeffsSample[i];
    }

    iter++;

    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    w.HamAndOvlpLanczos(walk, coeffsSample, ovlpSample, work, moreWork, alpha);
  }
  

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0][0]), gradError[0].size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[1][0]), gradError[1].size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[2][0]), gradError[2].size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[3][0]), gradError[3].size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(tauError[0]), tauError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, coeffs.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  for (int i = 0; i < 4; i++)
  {
    vector<double> b_size, r_x;
    blocking(b_size, r_x, gradError[i], tauError);
    rk[i] = corr_time(gradError.size(), b_size, r_x);
    S1[i] /= cumdeltaT;
  }

  double n_eff = commsize * (cumdeltaT * cumdeltaT) / cumdeltaT2;
  for (int i = 0; i < 4; i++) { 
    stddev[i] = sqrt(S1[i] * rk[i] / n_eff);
  }

  lanczosCoeffs = coeffs / commsize;

}

template<typename Wfn, typename Walker>
void getStochasticGradientContinuousTime(Wfn &w, Walker &walk, double &E0, double &stddev, Eigen::VectorXd &grad, double &rk, int niter, double targetError)
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
  //cout << walk << endl << endl;
  w.HamAndOvlp(walk, ovlp, ham, work);
  //cout << ham << "  " << ovlp << endl << endl;
  w.OverlapWithGradient(walk, ovlp, localdiagonalGrad);
  

  int nstore = 1000000 / commsize;
  int gradIter = min(nstore, niter);

  std::vector<double> gradError(gradIter * commsize, 0);
  std::vector<double> tauError(gradIter * commsize, 0);
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
    //if (commrank == 1) cout << walk.d<<"  "<<ham<<"  "<<Eloc<<"  "<<localdiagonalGrad.norm()<<endl;

    double ratio = deltaT / cumdeltaT;
    for (int i = 0; i < grad.rows(); i++)
    {
      diagonalGrad[i] += ratio * (localdiagonalGrad[i] - diagonalGrad[i]);
      grad[i] += ratio * (ham * localdiagonalGrad[i] - grad[i]);
      localdiagonalGrad[i] = 0.0;
    }

    Eloc = Eloc + deltaT * (ham - Eloc) / (cumdeltaT); //running average of energy

    S1 = S1 + deltaT * (ham - Elocold) * (ham - Eloc);

    if (iter < gradIter)
    {
      gradError[iter + commrank * gradIter] = ham;
      tauError[iter + commrank * gradIter] = deltaT;
    }
    iter++;

    //cout << "before  " << walk.d << endl;
    //cout << "hftype  " << w.getRef().hftype << endl;
    //cout << w.getRef().determinants[0] << endl;
    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
    //cout << "after  " << walk.d << endl;

    w.HamAndOvlp(walk, ovlp, ham, work);
    //cout << walk << endl;
    //cout << "ham  " <<  ham << "  ovlp  " << ovlp << endl << endl;
    
    w.OverlapWithGradient(walk, ovlp, localdiagonalGrad);

    if (abs(ovlp) > bestOvlp)
    {
      bestOvlp = abs(ovlp);
      bestDet = walk.getDet();
    }
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(tauError[0]), tauError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  //exit(0);
  //if (commrank == 0)
  vector<double> b_size, r_x;
  blocking(b_size, r_x, gradError, tauError);
  rk = corr_time(gradError.size(), b_size, r_x);
  //rk = calcTcorr(gradError);
  diagonalGrad /= (commsize);
  grad /= (commsize);
  E0 = Eloc / commsize;
  grad = grad - E0 * diagonalGrad;

  double n_eff = commsize * (cumdeltaT * cumdeltaT) / cumdeltaT2;
  S1 /= cumdeltaT;
  stddev = sqrt((S1 * rk / n_eff));
/*
  cout << "sample variance: " << S1 << endl;
  cout << "sample size: " << n_eff << endl;
  cout << "rk: " << rk << endl;
*/
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

class MetHelper
{
  public: 
  int norbs;
  vector<double> a, b; //all occ alpha/beta orbs of cdet
  vector<double> sa, sb; //all singly occ alpha/beta orbs of cdet
  vector<double> ua, ub; //all unocc alpha/beta orbs of cdet

  MetHelper(Determinant &d)
  {
    norbs = Determinant::norbs;
    for (int i = 0; i < norbs; i++)
    {
      bool alpha = d.getoccA(i);
      bool beta = d.getoccB(i);
      if (alpha) a.push_back(i);
      else ua.push_back(i);
      if (beta) b.push_back(i);
      else ub.push_back(i);
      if (alpha && !beta) sa.push_back(i);
      if (beta && !alpha) sb.push_back(i);
    }
  }

  double TotalMoves()
  {
    return (double) (a.size() * ua.size() + b.size() * ub.size() + sa.size() * sb.size());
  }
  double AlphaMoves()
  {
    return (double) (a.size() * ua.size());
  }
  double BetaMoves()
  {
    return (double) (b.size() * ub.size());
  }
  double DoubleMoves()
  {
    return (double) (sa.size() * sb.size());
  }
};

template<typename Wfn, typename Walker>
void getStochasticGradientMetropolis(Wfn &w, Walker &walk, double &E0, double &stddev, Eigen::VectorXd &grad, double &rk, int niter)
{
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1), std::ref(generator));

  workingArray work;
  stddev = 0.0;
  E0 = 0.0;
  int iter = 0;
  double S1 = 0.0, Eloc = 0.0, ovlp = 0.0;
  int numVars = grad.rows();
  grad.setZero();
  VectorXd grad_ratio_bar = VectorXd::Zero(numVars);
  VectorXd grad_ratio = VectorXd::Zero(numVars);
  int nstore = 1000000 / commsize;
  int gradIter = min(nstore, niter);
  std::vector<double> gradError(gradIter * commsize, 0);
  double bestOvlp = 0.0;
  Determinant bestDet = walk.getDet();
  enum Move
  {
    Amove,
    Bmove,
    Dmove
  };
  bool calcEloc = true;
  int nAMoves = 0;
  /*
  std::vector<double> e, t;
  double weight;
  */
  while (iter < niter)
  {
    if (calcEloc)
    {
      grad_ratio.setZero();
      w.HamAndOvlp(walk, ovlp, Eloc, work);
      w.OverlapWithGradient(walk, ovlp, grad_ratio);
    }
    for (int i = 0; i < grad.rows(); i++)
    {
      grad_ratio_bar[i] += grad_ratio[i];
      grad[i] += grad_ratio[i] * Eloc;;
    }
    double E0old = E0;
    E0 = E0 + (Eloc - E0) / (iter + 1);
    S1 = S1 + (Eloc - E0old) * (Eloc - E0);
    if (iter < gradIter)
    {
      gradError[iter + commrank * gradIter] = Eloc;
    }
    Determinant cdet = walk.getDet();  
    Determinant pdet = cdet;
    Move move;
    MetHelper C(cdet);
    double P_a = C.AlphaMoves() / C.TotalMoves();
    double P_b = C.BetaMoves() / C.TotalMoves();
    double P_d = C.DoubleMoves() / C.TotalMoves();
    double rand = random();
    int orb1, orb2;
    double pdetOvercdet;
    if (rand < P_a) 
    {
      move = Amove;
      orb1 = C.a[(int) (random() * C.a.size())];
      orb2 = C.ua[(int) (random() * C.ua.size())];
      pdet.setoccA(orb1, false);
      pdet.setoccA(orb2, true);
      pdetOvercdet = w.getOverlapFactor(2*orb1, 0, 2*orb2, 0, walk, false);
    }
    else if (rand < (P_b + P_a)) 
    {
      move = Bmove;
      orb1 = C.b[(int) (random() * C.b.size())];
      orb2 = C.ub[(int) (random() * C.ub.size())];
      pdet.setoccB(orb1, false);
      pdet.setoccB(orb2, true);
      pdetOvercdet = w.getOverlapFactor(2*orb1+1, 0, 2*orb2+1, 0, walk, false);
    }
    else if (rand < (P_d + P_a + P_b))
    {
      move = Dmove;
      orb1 = C.sa[(int) (random() * C.sa.size())];
      orb2 = C.sb[(int) (random() * C.sb.size())];
      pdet.setoccA(orb1, false);
      pdet.setoccB(orb2, false);
      pdet.setoccB(orb1, true);
      pdet.setoccA(orb2, true);
      pdetOvercdet = w.getOverlapFactor(2*orb1, 2*orb2+1, 2*orb2, 2*orb1+1, walk, false);
    }
    MetHelper P(pdet);
    double T_C = 1.0 / C.TotalMoves();
    double T_P = 1.0 / P.TotalMoves();
    double P_pdetOvercdet = pow(pdetOvercdet, 2.0);
    double accept = min(1.0, (T_P * P_pdetOvercdet) / T_C);
    if (random() < accept)
    {
      /*
      e.push_back(Eloc);
      t.push_back(weight);
      weight = 1.0;
      */
      nAMoves += 1;
      calcEloc = true;
      if (move == Amove)
      {
        walk.update(orb1, orb2, 0, w.getRef(), w.getCorr());
      }
      else if (move == Bmove)
      {
        walk.update(orb1, orb2, 1, w.getRef(), w.getCorr());
      }
      else if (move == Dmove)
      {
        walk.update(orb1, orb2, 0, w.getRef(), w.getCorr());
        walk.update(orb2, orb1, 1, w.getRef(), w.getCorr());
      }
    }
    else
    {
      calcEloc = false;
      //weight += 1.0;
    }
    iter++;
    if (abs(ovlp) > bestOvlp)
    {
      bestOvlp = abs(ovlp);
      bestDet = walk.getDet();
    }
  }
  grad_ratio_bar /= niter;
  grad /= niter;
  S1 /= niter;
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad_ratio_bar[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &E0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nAMoves, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
/*
//pseudo weights
  vector<double> C;
  corr_func(C, e, t);
  cout << "\t Pseudo data" << endl;
  cout << "\t mean: " << average(e, t) << endl;
  double tau = corr_time(C); 
  cout << "\t autocorrelation time: " << tau << endl;
  cout << "\t sample variance: " << variance(e,t) << endl;
  cout << "\t n_eff: " << n_eff(t) << endl;
  cout << "\t average variance: " << (variance(e, t) * tau / n_eff(t)) << endl;
  ofstream file("pseudo.txt");
  for (int i = 0; i < e.size(); i++)
  {
    file << e[i] << " \t" << t[i] << endl;
  }
  file.close();
*/
//blocking
  vector<double> b_size, r_x;
  blocking(b_size, r_x, gradError);
  rk = corr_time(gradError.size(), b_size, r_x);
  //write_block(b_size, r_x);
/*
  cout << "block rk: " << rk << endl;
//correlation function
  vector<double> c;
  corr_func(c, gradError);
  rk = corr_time(c);
  cout << "brute force rk: " << rk << endl;
  write_corr_func(c);
  cout << "sample variance: " << S1 << endl;
  cout << "sample size: " << niter * commsize << endl;
  cout << "Fraction of accepted moves: " << fracAmoves << endl;
*/
  grad_ratio_bar /= (commsize);
  grad /= (commsize);
  E0 /= commsize;
  grad = (grad - E0 * grad_ratio_bar);
  double n = niter * commsize;
  double fracAmoves = ((double) nAMoves)/ n;
  stddev = sqrt(S1 * rk / n);
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

template<typename Wfn, typename Walker>
void getStochasticGradientHessianContinuousTime(Wfn &w, Walker& walk, double &E0, double &stddev, oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE, VectorXd &grad, MatrixXd& Hessian, MatrixXd& Smatrix, double &rk, int niter, double targetError)
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
  std::vector<double> tauError(gradIter*commsize, 0);
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

    S1 = S1 + deltaT * (ham - Elocold) * (ham - Eloc);

    if (iter < gradIter)
    {
      gradError[iter + commrank*gradIter] = ham;
      tauError[iter + commrank * gradIter] = deltaT;
    }
    iter++;

    walk.updateWalker(w.getRef(), w.getCorr(), excitation1[nextDet], excitation2[nextDet]);

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
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(tauError[0]), tauError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(diagonalGrad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Hessian(0,0)), Hessian.rows()*Hessian.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Smatrix(0,0)), Smatrix.rows()*Smatrix.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  vector<double> b_size, r_x;
  blocking(b_size,r_x,gradError,tauError);
  rk = corr_time(gradError.size(), b_size, r_x);
  //rk = calcTcorr(gradError);

  diagonalGrad /= (commsize);
  grad /= (commsize);
  E0 = Eloc / commsize;
  grad = grad - E0 * diagonalGrad;
  Hessian = Hessian/(commsize);
  Smatrix = Smatrix/(commsize);

  S1 /= cumdeltaT
  double n_eff = commsize * (cumdeltaT * cumdeltaT) / cumdeltaT2;
  stddev = sqrt(S1 * rk / n_eff);
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

template<typename Wfn, typename Walker>
void getStochasticGradientMetricContinuousTime(Wfn &w, Walker& walk, double &E0, double &stddev, VectorXd &grad, VectorXd &H, DirectMetric &S, double &rk, int niter, double targetError)
{
  //cout << "a" << endl;
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;

  auto random = std::bind(std::uniform_real_distribution<double>(0, 1),std::ref(generator));
  
  workingArray work;

  stddev = 1.e4;
  int iter = 0;
  double M1 = 0., S1 = 0., Eavg = 0.;
  double Eloc = 0., ovlp = 0.;
  int numVars = grad.rows();
  H = VectorXd::Zero(numVars + 1);
  VectorXd grad_ratio_bar = VectorXd::Zero(numVars);
  VectorXd grad_ratio = VectorXd::Zero(numVars);

  double bestOvlp = 0.0;
  Determinant bestDet = walk.getDet();

  E0 = 0.0;
  w.HamAndOvlp(walk, ovlp, Eloc, work);
  w.OverlapWithGradient(walk, ovlp, grad_ratio);

  int nstore = 1000000/commsize;
  int gradIter = min(nstore, niter);

  std::vector<double> gradError(gradIter * commsize, 0);
  std::vector<double> tauError(gradIter * commsize, 0);
  double cumdeltaT = 0., cumdeltaT2 = 0.;

  S.T.setZero(niter);
  S.Vectors.setZero(numVars + 1, niter);

/*
  char sfile[50];
  sprintf(sfile, "./Metric/sVectors%i.bin", commrank);
  ofstream s(sfile, ios::binary);
  char tfile[50];
  sprintf(tfile, "./T/%i.bin", commrank);
  ofstream t(tfile, ios::binary); 
*/

  //cout << "b" << endl;
  while (iter < niter && stddev > targetError)
  {
    //cout << "i" << endl;

    double cumovlpRatio = 0;
    //when using uniform probability 1./numConnection * max(1, pi/pj)
    for (int i = 0; i < work.nExcitations; i++)
    {
      cumovlpRatio += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumovlpRatio;
    }
    //cout << "ii" << endl;

    //double deltaT = -log(random())/(cumovlpRatio);
    double deltaT = 1.0 / (cumovlpRatio);
    double nextDetRandom = random() * cumovlpRatio;
    int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations), nextDetRandom) - work.ovlpRatio.begin();

    cumdeltaT += deltaT;
    cumdeltaT2 += deltaT * deltaT;

    double E0old = E0;

    grad_ratio_bar += deltaT * (grad_ratio - grad_ratio_bar) / (cumdeltaT);
    grad += deltaT * (grad_ratio * Eloc - grad) / (cumdeltaT); //running average of grad
    E0 += deltaT * (Eloc - E0) / (cumdeltaT);       //running average of energy
    //cout << "iii" << endl;

    //Smatrix.block(1, 1, grad.rows(), grad.rows()) += deltaT * (localdiagonalGrad * localdiagonalGrad.transpose() - Smatrix.block(1, 1, grad.rows(), grad.rows())) / cumdeltaT;
    //Smatrix.block(0, 1, 1, grad.rows()) += deltaT * (localdiagonalGrad.transpose() - Smatrix.block(0, 1, 1, grad.rows())) / cumdeltaT;
    //Smatrix.block(1, 0, grad.rows(), 1) += deltaT * (localdiagonalGrad - Smatrix.block(1, 0, grad.rows(), 1)) / cumdeltaT;
    VectorXd appended(numVars + 1);
    appended << 1.0, grad_ratio;
/*
    t.write((char *)&deltaT, sizeof(double));
    s.write((char *)appended.data(), appended.size() * sizeof(double));
*/

    //cout << "iiii" << endl;
    S.T(iter) = deltaT;
    S.Vectors.col(iter) = appended;

    //Smatrix = (1 - deltaT / cumdeltaT) * Smatrix;
    //localdiagonalGrad = pow(deltaT/cumdeltaT, 0.5) * localdiagonalGrad;
    //VectorXd appended(grad.rows() + 1);
    //appended << pow(deltaT/cumdeltaT, 0.5), localdiagonalGrad;
    //Smatrix.noalias() += appended * appended.transpose();

    S1 += deltaT * (Eloc - E0old) * (Eloc - E0);

    if (iter < gradIter)
    {
      gradError[iter + commrank * gradIter] = Eloc;
      tauError[iter + commrank * gradIter] = deltaT;
    }
    iter++;

    walk.updateWalker(w.getRef(), w.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);

    grad_ratio.setZero();
    w.HamAndOvlp(walk, ovlp, Eloc, work);
    w.OverlapWithGradient(walk, ovlp, grad_ratio);
    //cout << "iiiii" << endl;

    if (abs(ovlp) > bestOvlp) {
      bestOvlp = abs(ovlp);
      bestDet = walk.getDet();
    }
  }
  //cout << "c" << endl;
  
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(gradError[0]), gradError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(tauError[0]), tauError.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad_ratio_bar[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &E0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  vector<double> r_x, b_size;
  blocking(b_size, r_x, gradError, tauError);
  rk = corr_time(gradError.size(), b_size, r_x);
  //write_block(b_size, r_x);

  grad_ratio_bar /= (commsize);
  grad /= (commsize);
  E0 /= commsize;

  grad = grad - E0 * grad_ratio_bar;
  
  VectorXd appended(numVars);
  appended = grad_ratio_bar - schd.stepsize * grad;
  H << 1.0, appended;

  S1 /= cumdeltaT;
  double n_eff = commsize * (cumdeltaT * cumdeltaT) / cumdeltaT2;
  stddev = sqrt(S1 * rk / n_eff);
#ifndef SERIAL
  MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  if (commrank == 0) {
    char file[5000];
    sprintf(file, "BestDeterminant.txt");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << bestDet;
  }
/*
  s.close();
  t.close();
*/
  //cout << "d" << endl;
}

template <typename Wfn, typename Walker>
class getGradientWrapper
{
 public:
  Wfn &w;
  Walker &walk;
  int stochasticIter;
  bool ctmc;
  getGradientWrapper(Wfn &pw, Walker &pwalk, int niter, bool pctmc) : w(pw), walk(pwalk)
  {
    stochasticIter = niter;
    ctmc = pctmc;
  };

  void getGradient(VectorXd &vars, VectorXd &grad, double &E0, double &stddev, double &rt, bool deterministic)
  {
    w.updateVariables(vars);
    w.initWalker(walk);
    if (!deterministic)
    {
      if (ctmc)
      {
        getStochasticGradientContinuousTime(w, walk, E0, stddev, grad, rt, stochasticIter, 0.5e-3);
      }
      else
      {
        getStochasticGradientMetropolis(w, walk, E0, stddev, grad, rt, stochasticIter);
      }
    }
    else
    {
      stddev = 0.0;
      rt = 1.0;
      getGradientDeterministic(w, walk, E0, grad);
    }
    w.writeWave();
  };
  
  void getMetric(VectorXd &vars, VectorXd &grad, VectorXd &H, DirectMetric &S, double &E0, double &stddev, double &rt, bool deterministic)
  {
      w.updateVariables(vars);
      w.initWalker(walk);
      if (!deterministic)
      {
        getStochasticGradientMetricContinuousTime(w, walk, E0, stddev, grad, H, S, rt, stochasticIter, 0.5e-3);
      }
      else
      {
        stddev = 0.0;
      	rt = 1.0;
      	getGradientMetricDeterministic(w, walk, E0, grad, H, S);
      }
      w.writeWave();
  };

  void getHessian(VectorXd &vars, VectorXd &grad, MatrixXd &Hessian, MatrixXd &smatrix, double &E0, double &stddev, oneInt &I1, twoInt &I2, twoIntHeatBathSHM &I2hb, double &coreE, double &rt, bool deterministic)
  {
    if (!deterministic)
    {
      w.updateVariables(vars);
      w.initWalker(walk);
      getStochasticGradientHessianContinuousTime(w, walk, E0, stddev, I1, I2, I2hb, coreE, grad, Hessian, Smatrix, rt, stochasticIter, 0.5e-3);
    }
    else
    {
      w.updateVariables(vars);
      w.initWalker(walk);
      stddev = 0.0;
      rt = 1.0;
      getGradientHessianDeterministic(w, walk, E0, nalpha, nbeta, norbs, I1, I2, I2hb, coreE, grad, Hessian, Smatrix)
    }
    w.writeWave();
  };
};
#endif
