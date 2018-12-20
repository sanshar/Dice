#ifndef CTMC_HEADER_H
#define CTMC_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "Determinants.h"
#include "workingArray.h"
#include "statistics.h"
#include "sr.h"
#include "global.h"
#include "evaluateE.h"
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

template<typename Wfn, typename Walker>
class ContinuousTime
{
  public:
  Wfn *w;
  Walker *walk;
  long numVars;
  int norbs, nalpha, nbeta;
  workingArray work;
  double T, Eloc, ovlp;
  double S1, oldEnergy;
  double cumT, cumT2;
  VectorXd grad_ratio;
  int nsample;
  Statistics Stats; //this is only used to calculate autocorrelation length
  Determinant bestDet;
  double bestOvlp;
  
  double random()
  {
    uniform_real_distribution<double> dist(0,1);
    return dist(generator);
  }
    
  ContinuousTime(Wfn &_w, Walker &_walk, int niter) : w(&_w), walk(&_walk)
  {
    nsample = min(niter, 200000);
    numVars = w->getNumVariables();
    norbs = Determinant::norbs;
    nalpha = Determinant::nalpha;
    nbeta = Determinant::nbeta;
    bestDet = walk->getDet();
    cumT = 0.0, cumT2 = 0.0, S1 = 0.0, oldEnergy = 0.0, bestOvlp = 0.0; 
  }

  void LocalEnergy()
  {
    Eloc = 0.0, ovlp = 0.0;
    w->HamAndOvlp(*walk, ovlp, Eloc, work);
  }

  void MakeMove()
  {
    double cumOvlp = 0.0;
    for (int i = 0; i < work.nExcitations; i++)
    {
      cumOvlp += abs(work.ovlpRatio[i]);
      work.ovlpRatio[i] = cumOvlp;
    }
    T = 1.0 / cumOvlp;
    double nextDetRand = random() * cumOvlp;
    int nextDet = lower_bound(work.ovlpRatio.begin(), work.ovlpRatio.begin() + work.nExcitations, nextDetRand) - work.ovlpRatio.begin();
    cumT += T;
    cumT2 += T * T;
    walk->updateWalker(w->getRef(), w->getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
  }

  void UpdateBestDet()
  {
    if (abs(ovlp) > bestOvlp)
    {
      bestOvlp = abs(ovlp);
      bestDet = walk->getDet();
    }
  }
  
  void FinishBestDet()
  {
    if (commrank == 0)
    {
      char file[50];
      sprintf(file, "BestDeterminant.txt");
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << bestDet;
    }
  }

  void UpdateEnergy(double &Energy)
  {
    oldEnergy = Energy;
    Energy += T * (Eloc - Energy) / cumT;
    S1 += T * (Eloc - oldEnergy) * (Eloc - Energy);
    if (Stats.X.size() < nsample)
    {
      Stats.push_back(Eloc, T);
    }
  }
  
  void FinishEnergy(double &Energy, double &stddev, double &rk)
  {
    Stats.Block();
    rk = Stats.BlockCorrTime();
/*
    if (commrank == 0)
    {
      Stats.WriteBlock();
      cout << "Block rk:\t" << rk << endl;
    }
    Stats.CorrFunc();
    rk = Stats.IntCorrTime();
    if (commrank == 0)
    {
      Stats.WriteCorrFunc();
      cout << "CorrFunc rk:\t" << rk << endl;
    }
    rk = calcTcorr(Stats.X);
    if (commrank == 0)
    {
      cout << "OldCorrFunc rk:\t" << rk << endl;
    }
*/
    S1 /= cumT;
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &Energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &S1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &rk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &cumT, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &cumT2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Energy /= commsize;
    S1 /= commsize;
    rk /= commsize;
    cumT /= commsize;
    cumT2 /= commsize;
#endif
    double neff = commsize * (cumT * cumT) / cumT2;
    stddev = sqrt(rk * S1 / neff);
  }

  void LocalGradient()
  {
    grad_ratio.setZero(numVars);
    w->OverlapWithGradient(*walk, ovlp, grad_ratio);
  }
  
  void UpdateGradient(VectorXd &grad, VectorXd &grad_ratio_bar)
  {
    grad_ratio_bar += T * (grad_ratio - grad_ratio_bar) / cumT;
    grad += T * (grad_ratio * Eloc - grad) / cumT;
  }

  void FinishGradient(VectorXd &grad, VectorXd &grad_ratio_bar, const double &Energy)
  {
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, (grad_ratio_bar.data()), grad_ratio_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, (grad.data()), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    grad /= commsize;
    grad_ratio_bar /= commsize;
#endif
    grad = (grad - Energy * grad_ratio_bar);
  }

  void UpdateSR(DirectMetric &S)
  {
    VectorXd appended(numVars + 1);
    appended << 1.0, grad_ratio;
    S.Vectors.push_back(appended);
    S.T.push_back(T);
  }
  
  void FinishSR(const VectorXd &grad, const VectorXd &grad_ratio_bar, VectorXd &H)
  {
    H.setZero(grad.rows() + 1);
    VectorXd appended(grad.rows());
    appended = grad_ratio_bar - schd.stepsize * grad;
    H << 1.0, appended;
  }
};
#endif
