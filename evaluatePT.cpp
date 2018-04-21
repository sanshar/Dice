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

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;
using namespace Eigen;



//<psi|H|psi>/<psi|psi> = <psi|d> <d|H|psi>/<psi|d><d|psi>
double evaluateScaledEDeterministic(Wfn& w, double& lambda, double& unscaledE0,
				    int& nalpha, int& nbeta, int& norbs,
				    oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				    double& coreE) {

  vector<vector<int> > alphaDets, betaDets;
  comb(norbs, nalpha, alphaDets);
  comb(norbs, nbeta , betaDets);
  std::vector<Determinant> allDets;
  for (int a=0; a<alphaDets.size(); a++)
    for (int b=0; b<betaDets.size(); b++) {
      Determinant d;
      for (int i=0; i<alphaDets[a].size(); i++)
	d.setoccA(alphaDets[a][i], true);
      for (int i=0; i<betaDets[b].size(); i++)
	d.setoccB(betaDets[b][i], true);
      allDets.push_back(d);
    }

  alphaDets.clear(); betaDets.clear();
  
  double E=0, ovlp=0;
  double unscaledE = 0;
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    double Eloc=0, ovlploc=0; 
    Walker walk(allDets[d]);
    walk.initUsingWave(w);
    w.HamAndOvlp(walk, ovlploc, Eloc, I1, I2, I2hb, coreE);
    E          += ((1-lambda)*Eloc + lambda*allDets[d].Energy(I1, I2, coreE))*ovlploc*ovlploc;
    unscaledE  += Eloc*ovlploc*ovlploc;
    ovlp       += ovlploc*ovlploc;
  }
  allDets.clear();

  double Ebkp=E, obkp = ovlp, unscaledEbkp = unscaledE;
  int size = 1;
#ifndef SERIAL
  MPI_Allreduce(&Ebkp, &E,     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&unscaledEbkp, &unscaledE,     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&obkp, &ovlp,  size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  unscaledE0 = unscaledE/ovlp; 
  return E/ovlp;
}




//<wave|MoDet>
double evaluateOvlpWithMoDet(Wfn& w, MoDeterminant& moDet, int& nalpha, int& nbeta, int& norbs,
			     oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double& coreE) {

  vector<vector<int> > alphaDets, betaDets;
  comb(norbs, nalpha, alphaDets);
  comb(norbs, nbeta , betaDets);
  std::vector<Determinant> allDets;
  for (int a=0; a<alphaDets.size(); a++)
    for (int b=0; b<betaDets.size(); b++) {
      Determinant d;
      for (int i=0; i<alphaDets[a].size(); i++)
	d.setoccA(alphaDets[a][i], true);
      for (int i=0; i<betaDets[b].size(); i++)
	d.setoccB(betaDets[b][i], true);
      allDets.push_back(d);
    }

  alphaDets.clear(); betaDets.clear();
  
  double A=0, B=0, C=0, ovlp=0;
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    double Eloc=0, ovlploc=0; 
    Walker walk(allDets[d]);
    walk.initUsingWave(w);
    double Ei = allDets[d].Energy(I1, I2, coreE);
    w.HamAndOvlp(walk, ovlploc, Eloc, I1, I2, I2hb, coreE);

    double moOvlp = moDet.Overlap(allDets[d]);
    ovlp += ovlploc*moOvlp;
  }
  allDets.clear();

  double obkp = ovlp;
  int size = 1;
#ifndef SERIAL
  MPI_Allreduce(&obkp, &ovlp,  size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  return ovlp;
}


//<psi|H|psi>/<psi|psi> = <psi|d> <d|H|psi>/<psi|d><d|psi>
double evaluatePTDeterministic(Wfn& w, double& E0, int& nalpha, int& nbeta, int& norbs,
			      oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double& coreE) {

  vector<vector<int> > alphaDets, betaDets;
  comb(norbs, nalpha, alphaDets);
  comb(norbs, nbeta , betaDets);
  std::vector<Determinant> allDets;
  for (int a=0; a<alphaDets.size(); a++)
    for (int b=0; b<betaDets.size(); b++) {
      Determinant d;
      for (int i=0; i<alphaDets[a].size(); i++)
	d.setoccA(alphaDets[a][i], true);
      for (int i=0; i<betaDets[b].size(); i++)
	d.setoccB(betaDets[b][i], true);
      allDets.push_back(d);
    }

  alphaDets.clear(); betaDets.clear();
  
  double A=0, B=0, C=0, ovlp=0;
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    double Eloc=0, ovlploc=0; 
    Walker walk(allDets[d]);
    walk.initUsingWave(w);
    double Ei = allDets[d].Energy(I1, I2, coreE);
    w.HamAndOvlp(walk, ovlploc, Eloc, I1, I2, I2hb, coreE);

    double ovlp2 = ovlploc*ovlploc;
    A    -= pow(Eloc-E0, 2)*ovlp2/(Ei-E0);
    B    += (Eloc-E0)*ovlp2/(Ei-E0);
    C    += ovlp2/(Ei-E0);
    ovlp += ovlp2;
  }
  allDets.clear();

  double obkp = ovlp;
  int size = 1;
#ifndef SERIAL
  MPI_Allreduce(&obkp, &ovlp,  size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  double Abkp=A/ovlp;
  double Bbkp=B/ovlp, Cbkp = C/ovlp;

#ifndef SERIAL
  MPI_Allreduce(&Abkp, &A,     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Bbkp, &B,     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&Cbkp, &C,     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (commrank == 0) cout << A<<"  "<<B*B/C<<"  "<<A+B*B/C<<"  "<<ovlp<<endl;
  return A + B*B/C;
}


//<psi|H|psi>/<psi|psi> = <psi|d> <d|H|psi>/<psi|d><d|psi>
double evaluatePTStochastic(CPSSlater& w, double& E0, int& nalpha, int& nbeta, int& norbs,
			    oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
			    double& coreE, double& stddev,
			    int niter, double targetError) {


  //initialize the walker
  Determinant d;
  for (int i=0; i<nalpha; i++)
    d.setoccA(i, true);
  for (int j=0; j<nbeta; j++)
    d.setoccB(j, true);
  Walker walk(d);
  walk.initUsingWave(w);


  stddev = 1.e4;
  int iter = 0;
  double M1=0., S1 = 0.;
  double A=0, B=0, C=0;
  double ham=0., ovlp =0.;
  double scale = 1.0;

  double rk = 1.;
  w.HamAndOvlp(walk, ovlp, ham, I1, I2, I2hb, coreE); 
  double Ei = walk.d.Energy(I1, I2, coreE);

  int gradIter = min(niter, 100000);
  std::vector<double> gradError(gradIter, 0);
  bool reset = true;

  /*
  while (iter <niter && stddev >targetError) {
    if (iter == 100 && reset) {
      iter = 0;
      reset = false;
      M1 = 0.; S1=0.;
      A=0; B=0; C=0;
      walk.initUsingWave(w, true);
    }

    double ovlp2= ovlp*ovlp;
    A   =   A  -  (pow(Eloc-E0, 2)/(Ei-E0) - A)/(iter+1);
    B   =   B  +  (pow(Eloc-E0, 2)/(Ei-E0) - A)/(iter+1);
    C   =   C  +  (pow(Eloc-E0, 2)/(Ei-E0) - A)/(iter+1);

    Eloc       = Eloc        + (ham - Eloc)/(iter+1);     //running average of energy
    ElocScaled = ElocScaled  + ((1-lambda)*ham+lambda*Ei - ElocScaled)/(iter+1);

    double Mprev = M1;
    M1 = Mprev + (ham - Mprev)/(iter+1);
    if (iter != 0)
      S1 = S1 + (ham - Mprev)*(ham - M1);

    if (iter < gradIter)
      gradError[iter] = ham;

    iter++;
    
    if (iter == gradIter-1) {
      rk = calcTcorr(gradError);
    }
    
    bool success = walk.makeMove(w);
    //bool success = walk.makeCleverMove(w, I1);
    //cout <<iter <<"  "<< walk.d<<"  "<<Eloc<<endl;
    //if (iter == 50) exit(0);
    if (success) {
      w.HamAndOvlp(walk, ovlp, ham, I1, I2, coreE); 
      Ei = walk.d.Energy(I1, I2, coreE);
    }
    
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ElocScaled, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  unscaledE = Eloc/commsize;

  stddev = sqrt(S1*rk/(niter-1)/niter/commsize) ;
#ifndef SERIAL
  MPI_Bcast(&stddev    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  */
  return 0.;
}


//<psi|H|psi>/<psi|psi> = <psi|d> <d|H|psi>/<psi|d><d|psi>
double evaluateScaledEStochastic(CPSSlater& w, double& lambda, double& unscaledE, 
				 int& nalpha, int& nbeta, int& norbs,
				 oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				 double& coreE, double& stddev,
				 int niter, double targetError) {


  //initialize the walker
  Determinant d;
  for (int i=0; i<nalpha; i++)
    d.setoccA(i, true);
  for (int j=0; j<nbeta; j++)
    d.setoccB(j, true);
  Walker walk(d);
  walk.initUsingWave(w);


  stddev = 1.e4;
  int iter = 0;
  double M1=0., S1 = 0., Eavg=0.;
  double Eloc=0.;
  double ElocScaled = 0.;
  double ham=0., ovlp =0.;
  double scale = 1.0;

  double E0 = 0.0, rk = 1.;
  w.HamAndOvlp(walk, ovlp, ham, I1, I2, I2hb, coreE); 
  double Ei = walk.d.Energy(I1, I2, coreE);

  int gradIter = min(niter, 100000);
  std::vector<double> gradError(gradIter, 0);
  bool reset = true;

  while (iter <niter && stddev >targetError) {
    if (iter == 100 && reset) {
      iter = 0;
      reset = false;
      M1 = 0.; S1=0.;
      Eloc = 0;
      ElocScaled = 0;
      walk.initUsingWave(w, true);
    }

    Eloc       = Eloc        + (ham - Eloc)/(iter+1);     //running average of energy
    ElocScaled = ElocScaled  + ((1-lambda)*ham+lambda*Ei - ElocScaled)/(iter+1);

    double Mprev = M1;
    M1 = Mprev + (ham - Mprev)/(iter+1);
    if (iter != 0)
      S1 = S1 + (ham - Mprev)*(ham - M1);

    if (iter < gradIter)
      gradError[iter] = ham;

    iter++;
    
    if (iter == gradIter-1) {
      rk = calcTcorr(gradError);
    }
    
    bool success = walk.makeMove(w);
    //bool success = walk.makeCleverMove(w, I1);
    //cout <<iter <<"  "<< walk.d<<"  "<<Eloc<<endl;
    //if (iter == 50) exit(0);
    if (success) {
      w.HamAndOvlp(walk, ovlp, ham, I1, I2, I2hb, coreE); 
      Ei = walk.d.Energy(I1, I2, coreE);
    }
    
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ElocScaled, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  unscaledE = Eloc/commsize;

  stddev = sqrt(S1*rk/(niter-1)/niter/commsize) ;
#ifndef SERIAL
  MPI_Bcast(&stddev    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  return ElocScaled/commsize;
}


double evaluatePTDeterministic2(Wfn& w, double& E0, int& nalpha, int& nbeta, int& norbs,
				oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, double& coreE) {

  vector<vector<int> > alphaDets, betaDets;
  comb(norbs, nalpha, alphaDets);
  comb(norbs, nbeta , betaDets);
  std::vector<Determinant> allDets;
  for (int a=0; a<alphaDets.size(); a++)
    for (int b=0; b<betaDets.size(); b++) {
      Determinant d;
      for (int i=0; i<alphaDets[a].size(); i++)
	d.setoccA(alphaDets[a][i], true);
      for (int i=0; i<betaDets[b].size(); i++)
	d.setoccB(betaDets[b][i], true);
      allDets.push_back(d);
    }

  alphaDets.clear(); betaDets.clear();

  SparseHam Ham;
  vector<vector<int> >& connections = Ham.connections;
  vector<vector<double> >& Helements = Ham.Helements;
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    connections.push_back(vector<int>(1, d));
    Helements.push_back(vector<double>(1, allDets[d].Energy(I1, I2, coreE)));

    for (int i=d+1; i<allDets.size(); i++) {
      if (allDets[d].connected(allDets[i])) {
	connections.rbegin()->push_back(i);
	Helements.rbegin()->push_back( Hij(allDets[d], allDets[i], I1, I2, coreE));
      }
    }
  }

  Hmult2 hmult(Ham);

  MatrixXx psi0 = MatrixXx::Zero(allDets.size(),1);
  MatrixXx diag = MatrixXx::Zero(allDets.size(),1);
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    double Eloc=0, ovlploc=0; 
    Walker walk(allDets[d]);
    walk.initUsingWave(w);
    double Ei = allDets[d].Energy(I1, I2, coreE);
    w.HamAndOvlp(walk, ovlploc, Eloc, I1, I2, I2hb, coreE);

    psi0(d,0) = ovlploc;
    diag(d,0) = Ei;
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &psi0(0,0),  psi0.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &diag(0,0),  diag.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  psi0 /= psi0.norm();

  MatrixXx Hpsi0 = MatrixXx::Zero(allDets.size(),1);
  hmult(&psi0(0,0), &Hpsi0(0,0));
#ifndef SERIAL
  MPI_Bcast(&Hpsi0(0,0),  Hpsi0.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  MatrixXx x0;
  if (commrank == 0) cout << psi0.adjoint()*Hpsi0<<endl;

  vector<double*> proj(1, &psi0(0,0));
  int index = 0;
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    Ham.connections[index].resize(1);
    Ham.Helements[index].resize(1);
    index++;
  }
  LinearSolver(hmult, E0, x0, Hpsi0, proj, 1.e-6, true);

  return 0;
}
