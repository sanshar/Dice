#ifndef DETERMINISTIC_HEADER_H
#define DETERMINISTIC_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include "Determinants.h"
#include "workingArray.h"
#include "statistics.h"
#include "sr.h"
#include "evaluateE.h"
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

template<typename Wfn, typename Walker>
class Deterministic
{
  public:
  Wfn w;
  Walker walk;
  long numVars;
  int norbs, nalpha, nbeta;
  vector<Determinant> allDets;
  workingArray work;
  double ovlp, Eloc;
  double Overlap;
  VectorXd grad_ratio;

  Deterministic(Wfn _w, Walker _walk) : w(_w), walk(_walk)
  {
    numVars = w.getNumVariables();
    norbs = Determinant::norbs;
    nalpha = Determinant::nalpha;
    nbeta = Determinant::nbeta;
    generateAllDeterminants(allDets, norbs, nalpha, nbeta);
    Overlap = 0.0;
  }
   
  void LocalEnergy(Determinant &D)
  {
    ovlp = 0.0, Eloc = 0.0;
    w.initWalker(walk, D);
    w.HamAndOvlp(walk, ovlp, Eloc, work, false);  
  }
  
  void UpdateEnergy(double &Energy)
  {
    Overlap += ovlp * ovlp;
    Energy += Eloc * ovlp * ovlp;
  }

  void FinishEnergy(double &Energy)
  {
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &(Overlap), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &(Energy), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    Energy /= Overlap;
  }

  void LocalGradient()
  {
    grad_ratio.setZero(numVars);
    w.OverlapWithGradient(walk, ovlp, grad_ratio);
  }

  void UpdateGradient(VectorXd &grad, VectorXd &grad_ratio_bar)
  {
    grad += grad_ratio * Eloc * ovlp * ovlp;
    grad_ratio_bar += grad_ratio * ovlp * ovlp;
  }

  void FinishGradient(VectorXd &grad, VectorXd &grad_ratio_bar, const double &Energy)
  {
#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, (grad_ratio_bar.data()), grad_ratio_bar.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, (grad.data()), grad_ratio.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    grad = (grad - Energy * grad_ratio_bar) / Overlap;
  }
  
  void UpdateSR(DirectMetric &S)
  {
    VectorXd appended(numVars);
    appended << 1.0, grad_ratio;
    S.Vectors.push_back(appended);
    S.T.push_back(ovlp * ovlp);
  }
  
  void FinishSR(const VectorXd &grad, const VectorXd &grad_ratio_bar, VectorXd &H)
  {
    H.setZero(numVars + 1);
    VectorXd appended(numVars);
    appended = grad_ratio_bar - schd.stepsize * grad;
    H << 1.0, appended;
  }
};
#endif
