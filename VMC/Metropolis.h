#ifndef MET_HEADER_H
#define MET_HEADER_H
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

enum Move
{
  Amove,
  Bmove,
  Dmove
};

class DetMoves
{
  public: 
  int norbs;
  vector<int> a, b; //all occ alpha/beta orbs of cdet
  vector<int> sa, sb; //all singly occ alpha/beta orbs of cdet
  vector<int> ua, ub; //all unocc alpha/beta orbs of cdet

  DetMoves(Determinant &d)
  {
    norbs = Determinant::norbs;
    for (int i = 0; i < norbs; i++)
    {
      bool alpha = d.getoccA(i);
      bool beta = d.getoccB(i);
      if (alpha)
        a.push_back(i);
      else
        ua.push_back(i);
      if (beta)
        b.push_back(i);
      else
        ub.push_back(i);
      if (alpha && !beta)
        sa.push_back(i);
      if (beta && !alpha)
        sb.push_back(i);
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
class Metropolis
{
  public:
  Wfn *w;
  Walker *walk;
  long numVars;
  int norbs, nalpha, nbeta;
  workingArray work;
  double Eloc, ovlp;
  double S1, oldEnergy;
  VectorXd grad_ratio;
  int nsample;
  Statistics Stats; //this is only used to calculate autocorrelation length
  //auto random = bind(uniform_real_distribution<double>dist(0,1), ref(generator));
  Determinant bestDet;
  double bestOvlp;
  bool calcEloc;
  double fracAmoves, n, nAMoves;
  
  double random()
  {
    uniform_real_distribution<double> dist(0,1);
    return dist(generator);
  }
  
  Metropolis(Wfn &_w, Walker &_walk, int niter) : w(&_w), walk(&_walk)
  {
    nsample = min(niter, 200000);
    numVars = w->getNumVariables();
    norbs = Determinant::norbs;
    nalpha = Determinant::nalpha;
    nbeta = Determinant::nbeta;
    bestDet = walk->getDet();
    S1 = 0.0, oldEnergy = 0.0, bestOvlp = 0.0, nAMoves = 0.0, n = 0.0;
    calcEloc = true;
  }
  
  void LocalEnergy()
  {
    if (calcEloc)
    {
      Eloc = 0.0, ovlp = 0.0;
      w->HamAndOvlp(*walk, ovlp, Eloc, work);
    }
  }

  void MakeMove()
  {
    if (abs(ovlp) > bestOvlp)
    {
      bestOvlp = abs(ovlp);
      bestDet = walk->getDet();
    }
    Determinant cdet = walk->getDet();  
    Determinant pdet = cdet;
    Move move;
    DetMoves C(cdet);
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
      pdetOvercdet = w->getOverlapFactor(2*orb1, 0, 2*orb2, 0, *walk, false);
    }
    else if (rand < (P_b + P_a)) 
    {
      move = Bmove;
      orb1 = C.b[(int) (random() * C.b.size())];
      orb2 = C.ub[(int) (random() * C.ub.size())];
      pdet.setoccB(orb1, false);
      pdet.setoccB(orb2, true);
      pdetOvercdet = w->getOverlapFactor(2*orb1+1, 0, 2*orb2+1, 0, *walk, false);
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
      pdetOvercdet = w->getOverlapFactor(2*orb1, 2*orb2+1, 2*orb2, 2*orb1+1, *walk, false);
    }
    DetMoves P(pdet);
    double T_C = 1.0 / C.TotalMoves();
    double T_P = 1.0 / P.TotalMoves();
    double P_pdetOvercdet = pdetOvercdet * pdetOvercdet;
    double accept = min(1.0, (T_P * P_pdetOvercdet) / T_C);
    if (random() < accept)
    {
      /*
      e.push_back(Eloc);
      t.push_back(weight);
      weight = 1.0;
      */
      nAMoves += 1.0;
      calcEloc = true;
      if (move == Amove)
      {
        walk->update(orb1, orb2, 0, w->getRef(), w->getCorr());
      }
      else if (move == Bmove)
      {
        walk->update(orb1, orb2, 1, w->getRef(), w->getCorr());
      }
      else if (move == Dmove)
      {
        walk->update(orb1, orb2, 0, w->getRef(), w->getCorr());
        walk->update(orb2, orb1, 1, w->getRef(), w->getCorr());
      }
    }
    else
    {
      calcEloc = false;
      //weight += 1.0;
    }
    n += 1.0;
  }

  void UpdateEnergy(double &Energy)
  {
    oldEnergy = Energy;
    Energy += (Eloc - Energy) / n;
    S1 += (Eloc - oldEnergy) * (Eloc - Energy);
    if (Stats.X.size() < nsample)
    {
      Stats.push_back(Eloc);
    }
  }
  
  void FinishEnergy(double &Energy, double &stddev, double &rk)
  {
    Stats.Block();
    rk = Stats.BlockCorrTime();
    if (commrank == 0)
    { 
      Stats.WriteBlock();
    }
    S1 /= n;
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &Energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &S1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &rk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &nAMoves, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Energy /= commsize;
    S1 /= commsize;
    rk /= commsize;
#endif
    n *= (double) commsize;
    stddev = sqrt(rk * S1 / n);
    fracAmoves = nAMoves / n;
    if (commrank == 0)
    {
      cout << "Fraction of accepted moves:\t" << fracAmoves << endl;
      char file[50];
      sprintf(file, "BestDeterminant.txt");
      std::ofstream ofs(file, std::ios::binary);
      boost::archive::binary_oarchive save(ofs);
      save << bestDet;
    }
  }

  void LocalGradient()
  {
    if (calcEloc)
    {
      grad_ratio.setZero(numVars);
      w->OverlapWithGradient(*walk, ovlp, grad_ratio);
    }
  }

  void UpdateGradient(VectorXd &grad, VectorXd &grad_ratio_bar)
  {
    grad_ratio_bar += grad_ratio;
    grad += grad_ratio * Eloc;
  }
  
  void FinishGradient(VectorXd &grad, VectorXd &grad_ratio_bar, const double &Energy)
  {
    grad /= n;
    grad_ratio_bar /= n;
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, (grad_ratio_bar.data()), grad_ratio_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, (grad.data()), grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    grad /= commsize;
    grad_ratio_bar /= commsize;
#endif
    grad = (grad - Energy * grad_ratio_bar);
  }
};
  
#endif
