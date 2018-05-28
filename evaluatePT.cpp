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
#include "Wfn.h"
#include <algorithm>
#include "integral.h"
#include "Determinants.h"
#include "Walker.h"
#include <boost/format.hpp>
#include <iostream>
#include "evaluateE.h"
#include "evaluatePT.h"
#include "Davidson.h"
#include "Hmult.h"
#include "global.h"
#include "input.h"

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

  VectorXd localGrad; bool doGradient = false;
  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;
  vector<double> Hij;

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
    double scale = 1.0, E0;
    Walker walk(allDets[d]);
    walk.initUsingWave(w);
    w.HamAndOvlpGradient(walk, ovlploc, Eloc, localGrad, I1, I2, I2hb, coreE,
			 ovlpRatio, excitation1, excitation2, Hij, doGradient);

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
  VectorXd localGrad; bool doGradient = false;
  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;
  vector<double> Hij;

  double A=0, B=0, C=0, ovlp=0;
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    double Eloc=0, ovlploc=0; 
    double scale = 1.0;
    Walker walk(allDets[d]);
    walk.initUsingWave(w);
    double Ei = allDets[d].Energy(I1, I2, coreE);
    w.HamAndOvlpGradient(walk, ovlploc, Eloc, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
			 excitation1, excitation2, Hij, doGradient);
    //w.HamAndOvlp(walk, ovlploc, Eloc, I1, I2, I2hb, coreE);

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
  if (commrank == 0) cout <<A<<"  "<< B<<"  "<<C<<"  "<<B*B/C<<endl;
  return A + B*B/C;
}


//<psi|H|psi>/<psi|psi> = <psi|d> <d|H|psi>/<psi|d><d|psi>
double evaluatePTDeterministicB(Wfn& w, double& E0, int& nalpha, int& nbeta, int& norbs,
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
  VectorXd localGrad; bool doGradient = false;
  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;
  vector<double> Hij;

  //if (commrank == 0) cout << allDets.size()<<endl;

  double A=0, B=0, C=0, ovlp=0;
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    double Eloc=0, ovlploc=0; 
    double scale = 1.0;
    Walker walk(allDets[d]);
    walk.initUsingWave(w);
    double Ei = allDets[d].Energy(I1, I2, coreE);

    double Aloc=0, Bloc=0, Cloc=0;
    vector<double> ovlpRatio; vector<size_t> excitation1, excitation2; bool doGradient=false;
    w.PTcontribution2ndOrder(walk, E0, I1, I2, I2hb, coreE, Aloc, Bloc, Cloc,
			     ovlpRatio, excitation1, excitation2, doGradient); 

    w.HamAndOvlpGradient(walk, ovlploc, Eloc, localGrad, I1, I2, I2hb, coreE, ovlpRatio,
			 excitation1, excitation2, Hij, doGradient);

    double ovlp2 = ovlploc*ovlploc;

    A    += Aloc*ovlp2;
    B    += Bloc*ovlp2;
    C    += Cloc*ovlp2;
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
  if (commrank == 0) cout <<A<<"  "<< B<<"  "<<C<<"  "<<B*B/C<<endl;
  return A + B*B/C;
}


double evaluatePTDeterministicC(Wfn& w, double& E0, int& nalpha, int& nbeta, int& norbs,
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

  SparseHam Ham0 = Ham;
  Hmult2 hmult(Ham0);
  VectorXd localGrad; bool doGradient = false;
  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;
  vector<double> Hij;

  MatrixXx psi0 = MatrixXx::Zero(allDets.size(),1);
  MatrixXx diag = MatrixXx::Zero(allDets.size(),1);
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    double Eloc=0, ovlploc=0; 
    Walker walk(allDets[d]);
    walk.initUsingWave(w);
    double Ei = allDets[d].Energy(I1, I2, coreE);
    double scale = 1.0, E0;
    w.HamAndOvlpGradient(walk, ovlploc, Eloc, localGrad, I1, I2, I2hb, coreE,
			 ovlpRatio, excitation1, excitation2, Hij, doGradient); 
    //w.HamAndOvlp(walk, ovlploc, Eloc, I1, I2, I2hb, coreE);

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
    Ham0.connections[index].resize(1);
    Ham0.Helements[index].resize(1);
    index++;
  }
  double PT = LinearSolver(hmult, E0, x0, Hpsi0, proj, 1.e-6, true);

  
#ifndef SERIAL
  MPI_Bcast(&x0(0,0),  x0.rows(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  if (commrank == 0) cout << x0.adjoint()*psi0<<endl;

  Hmult2 hmultfull(Ham);
  MatrixXx HHpsi0 = MatrixXx::Zero(allDets.size(), 1);
  MatrixXx HHpsi1 = MatrixXx::Zero(allDets.size(), 1);
  hmultfull(&x0(0,0), &HHpsi0(0,0));  //H x0
  hmult    (&x0(0,0), &HHpsi1(0,0));  //H0 x0

  if (commrank == 0) cout << HHpsi0.adjoint()*x0-HHpsi1.adjoint()*x0<<endl;

  return -PT;
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


  VectorXd localGrad; bool doGradient = false;
  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;
  vector<double> Hij;

  stddev = 1.e4;
  int iter = 0;
  double M1=0., S1 = 0., Eavg=0.;
  double Eloc=0.;
  double ElocScaled = 0.;
  double ham=0., ovlp =0.;
  double scale = 1.0;

  double E0 = 0.0, rk = 1.;
  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE,
		       ovlpRatio, excitation1, excitation2, Hij, doGradient); 
  //w.HamAndOvlp(walk, ovlp, ham, I1, I2, I2hb, coreE); 
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
    
    bool success = walk.makeCleverMove(w);
    //bool success = walk.makeCleverMove(w, I1);
    //cout <<iter <<"  "<< walk.d<<"  "<<Eloc<<endl;
    //if (iter == 50) exit(0);
    if (success) {
      w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE,
			   ovlpRatio, excitation1, excitation2, Hij, doGradient); 
      //w.HamAndOvlp(walk, ovlp, ham, I1, I2, I2hb, coreE); 
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




double evaluatePTStochasticMethodA(CPSSlater& w, double& E0, int& nalpha, int& nbeta, int& norbs,
				   oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				   double& coreE, double& stddev,
				   int niter, double& A, double& B, double& C) {


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
  A=0; B=0; C=0;
  double Aloc=0, Bloc=0, Cloc=0;
  double scale = 1.0;

  double rk = 1.;
  vector<double> ovlpRatio; vector<size_t> excitation1, excitation2; bool doGradient=false;
  w.PTcontribution2ndOrder(walk, E0, I1, I2, I2hb, coreE, Aloc, Bloc, Cloc,
		   ovlpRatio, excitation1, excitation2, doGradient); 

  int gradIter = min(niter, 100000);
  std::vector<double> gradError(gradIter, 0);
  bool reset = true;


  while (iter <niter ) {
    if (iter == 100 && reset) {
      iter = 0;
      reset = false;
      M1 = 0.; S1=0.;
      A=0; B=0; C=0;
      walk.initUsingWave(w, true);
      //Detmap.clear();
    }



    A   =   A  +  ( Aloc - A)/(iter+1);
    B   =   B  +  ( Bloc - B)/(iter+1);
    C   =   C  +  ( Cloc - C)/(iter+1);

    double Mprev = M1;
    M1 = Mprev + (Aloc - Mprev)/(iter+1);
    if (iter != 0)
      S1 = S1 + (Aloc - Mprev)*(Aloc - M1);

    if (iter < gradIter)
      gradError[iter] = Aloc;

    iter++;
    
    if (iter == gradIter-1) {
      rk = calcTcorr(gradError);
    }
    
    //bool success = walk.makeCleverMove(w);
    bool success = walk.makeMove(w);

    if (success) {
      ovlpRatio.clear(); excitation1.clear(); excitation2.clear();
      w.PTcontribution2ndOrder(walk, E0, I1, I2, I2hb, coreE, Aloc, Bloc, Cloc,
		   ovlpRatio, excitation1, excitation2, doGradient); 
    }
    
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  A = A/commsize;
  B = B/commsize;
  C = C/commsize;


  stddev = sqrt(S1*rk/(niter-1)/niter/commsize) ;
#ifndef SERIAL
  MPI_Bcast(&stddev    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  return A + B*B/C;
}



//A = sum_i <D_i|(H-E0)|Psi>/(Ei-E0) pi 
//where pi = <D_i|psi>**2/<psi|psi>
//this introduces a bias because there are determinants that have a near zero overlap with
//psi but make a non-zero contribution to A.
double evaluatePTStochasticMethodB(CPSSlater& w, double& E0, int& nalpha, int& nbeta, int& norbs,
				   oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				   double& coreE, double& stddev,
				   int niter, double& A, double& B, double& C) {


  //initialize the walker
  Determinant d;
  for (int i=0; i<nalpha; i++)
    d.setoccA(i, true);
  for (int j=0; j<nbeta; j++)
    d.setoccB(j, true);
  Walker walk(d);
  walk.initUsingWave(w);


  VectorXd localGrad; bool doGradient = false;
  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;
  vector<double> Hij;


  stddev = 1.e4;
  int iter = 0;
  double M1=0., S1 = 0.;
  A=0; B=0; C=0;
  double Aloc=0, Bloc=0, Cloc=0;
  double scale = 1.0, ham, ovlp;

  double rk = 1.;

  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE,
		       ovlpRatio, excitation1, excitation2, Hij, doGradient); 

  double Ei = walk.d.Energy(I1, I2, coreE);

  int gradIter = min(niter, 100000);
  std::vector<double> gradError(gradIter, 0);
  bool reset = true;

  while (iter <niter ) {
    if (iter == 20 && reset) {
      iter = 0;
      reset = false;
      M1 = 0.; S1=0.;
      A=0; B=0; C=0;
      walk.initUsingWave(w, true);
    }

    Aloc = -pow(ham-E0, 2)/(Ei-E0);
    Bloc = (ham-E0)/(Ei-E0);
    Cloc = 1./(Ei-E0);
    
    A   =   A  +  ( Aloc - A)/(iter+1);
    B   =   B  +  ( Bloc - B)/(iter+1);
    C   =   C  +  ( Cloc - C)/(iter+1);


    double Mprev = M1;
    M1 = Mprev + (Aloc - Mprev)/(iter+1);
    if (iter != 0)
      S1 = S1 + (Aloc - Mprev)*(Aloc - M1);

    if (iter < gradIter)
      gradError[iter] = Aloc;

    iter++;
    
    if (iter == gradIter-1) {
      rk = calcTcorr(gradError);
    }
    
    //bool success = walk.makeCleverMove(w);
    bool success = walk.makeMove(w);

    if (success) {
      w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, I1, I2, I2hb, coreE,
			   ovlpRatio, excitation1, excitation2, Hij, doGradient); 
      //w.HamAndOvlp(walk, ovlp, ham, I1, I2, I2hb, coreE); 
      Ei = walk.d.Energy(I1, I2, coreE);
    }
    
  }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  A = A/commsize;
  B = B/commsize;
  C = C/commsize;

  stddev = sqrt(S1*rk/(niter-1)/niter/commsize) ;
#ifndef SERIAL
  MPI_Bcast(&stddev    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  if (commrank == 0) cout << rk<<endl;

  return A + B*B/C;
}



//\sum_i <psi|(H0-E0)|D_j><D_j|H0-E0|Di>/(Ei-E0)/<Di|psi>  pi
//where pi = <D_i|psi>**2/<psi|psi>
double evaluatePTStochasticMethodC(CPSSlater& w, double& E0, int& nalpha, int& nbeta, int& norbs,
				   oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				   double& coreE, double& stddevA,double& stddevB, double& stddevC,
				   int niter, double& A, double& B, double& C) {

  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  //initialize the walker
  Determinant d;
  for (int i=0; i<nalpha; i++)
    d.setoccA(i, true);
  for (int j=0; j<nbeta; j++)
    d.setoccB(j, true);
  Walker walk(d);
  walk.initUsingWave(w);

  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;

  stddevA = 1.e4; stddevB = 1.e-4; stddevC = 1.e-4;
  int iter = 0;
  double M1=0., SA1 = 0., SB1=0., SC1=0.;
  A=0; B=0; C=0;
  double Aloc=0, Bloc=0, Cloc=0;
  double scale = 1.0;

  double rk = 1.;
  w.PTcontribution2ndOrder(walk, E0, I1, I2, I2hb, coreE, Aloc, Bloc, Cloc,
			   ovlpRatio, excitation1, excitation2, false); 
  
  int gradIter = min(niter, 1000);
  std::vector<double> gradError(gradIter, 0);
  bool reset = true;
  double cumdeltaT = 0., cumdeltaT2=0.;

  while (iter <niter ) {
    if (iter == 100 && reset) {
      iter = 0;
      reset = false;
      SA1=0.; SB1=0.; SC1=0.;
      A=0; B=0; C=0;
      walk.initUsingWave(w, true);
      cumdeltaT = 0; cumdeltaT2 = 0;
    }

    double cumovlpRatio = 0.;
    for (int i=0; i<ovlpRatio.size(); i++) {
      cumovlpRatio += abs(ovlpRatio[i]);
      ovlpRatio[i]  = cumovlpRatio;
    }

    double deltaT = 1.0/(cumovlpRatio);

    double select = random()*cumovlpRatio;
    int nextDet = std::distance(ovlpRatio.begin(), std::lower_bound(ovlpRatio.begin(), ovlpRatio.end(), 
				   select));

    cumdeltaT  += deltaT;
    cumdeltaT2 += deltaT*deltaT;
    double Aold = A, Bold=B, Cold=C;
    //if (commrank == 0) cout <<iter<<"  CCC "<< walk.d<<"  "<<Aloc<<"  "<<Bloc<<"  "<<Cloc<<endl;

    A   =   A  +  deltaT*( Aloc - A)/(cumdeltaT);
    B   =   B  +  deltaT*( Bloc - B)/(cumdeltaT);
    C   =   C  +  deltaT*( Cloc - C)/(cumdeltaT);

    SA1 = SA1 + (Aloc - Aold)*(Aloc - A);
    SB1 = SB1 + (Bloc - Bold)*(Bloc - B);
    SC1 = SC1 + (Cloc - Cold)*(Cloc - C);

    if (iter < gradIter)
      gradError[iter] = Bloc;

    iter++;
    
    if (iter == gradIter-1) {
      rk = calcTcorr(gradError);
    }

    //update the walker
    if (true) {

      int I = excitation1[nextDet]/2/norbs, A = excitation1[nextDet] - 2*norbs*I;
      if (I%2 == 0) walk.updateA(I/2, A/2, w);
      else walk.updateB(I/2, A/2, w);

      if (excitation2[nextDet] != 0) {
	int I = excitation2[nextDet]/2/norbs, A = excitation2[nextDet] - 2*norbs*I;
	if (I%2 == 0) walk.updateA(I/2, A/2, w);
	else walk.updateB(I/2, A/2, w);
      }
      ovlpRatio.clear(); excitation1.clear(); excitation2.clear();

      w.PTcontribution2ndOrder(walk, E0, I1, I2, I2hb, coreE, Aloc, Bloc, Cloc,
			       ovlpRatio, excitation1, excitation2, false); 
    }
    
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  A = A/commsize;
  B = B/commsize;
  C = C/commsize;

  stddevA = sqrt(SA1*rk/(niter-1)/niter/commsize) ;
  stddevB = sqrt(SB1*rk/(niter-1)/niter/commsize) ;
  stddevC = sqrt(SC1*rk/(niter-1)/niter/commsize) ;
#ifndef SERIAL
  MPI_Bcast(&stddevA    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&stddevB    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&stddevC    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  if (commrank == 0) cout << rk<<endl;
  return A + B*B/C;
}


double evaluatePTStochastic3rdOrder(CPSSlater& w, double& E0, int& nalpha, int& nbeta, int& norbs,
				   oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2hb, 
				   double& coreE, double& stddev,
				    int niter, double& A2, double& B, double& C, double& A3) {

  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  //initialize the walker
  Determinant d;
  for (int i=0; i<nalpha; i++)
    d.setoccA(i, true);
  for (int j=0; j<nbeta; j++)
    d.setoccB(j, true);
  Walker walk(d);
  walk.initUsingWave(w);

  vector<double> ovlpRatio;
  vector<size_t> excitation1, excitation2;

  stddev = 1.e4;
  int iter = 0;
  double M1=0., S1 = 0.;
  A2=0; B=0; C=0;A3=0;
  double A2loc=0, A3loc=0, Bloc=0, Cloc=0;
  double scale = 1.0;

  double rk = 1.;
  w.PTcontribution3rdOrder(walk, E0, I1, I2, I2hb, coreE, A2loc, Bloc, Cloc, A3loc,
			   ovlpRatio, excitation1, excitation2, false); 
  
  int gradIter = min(niter, 1000);
  std::vector<double> gradError(gradIter, 0);
  bool reset = true;
  double cumdeltaT = 0., cumdeltaT2=0.;


  while (iter <niter ) {
    if (iter == 100 && reset) {
      iter = 0;
      reset = false;
      M1 = 0.; S1=0.;
      A2=0; B=0; C=0;A3 = 0;
      walk.initUsingWave(w, true);
      cumdeltaT = 0; cumdeltaT2 = 0;
    }

    double cumovlpRatio = 0;
    for (int i=0; i<ovlpRatio.size(); i++) {
      //cumovlpRatio += min(1.0, ovlpRatio[i]);
      cumovlpRatio += abs(ovlpRatio[i]);
      ovlpRatio[i]  = cumovlpRatio;
    }

    //double deltaT = -log(random())/(cumovlpRatio);
    double deltaT = 1.0/(cumovlpRatio);
    double select = random()*cumovlpRatio;
    int nextDet = std::distance(ovlpRatio.begin(), std::lower_bound(ovlpRatio.begin(), ovlpRatio.end(), 
				   select));

    cumdeltaT  += deltaT;
    cumdeltaT2 += deltaT*deltaT;
    //if (commrank == 0) cout <<iter<<"  CCC "<< walk.d<<"  "<<Aloc<<"  "<<Bloc<<"  "<<Cloc<<endl;
    double A3old = A3;

    A2   =   A2  +  deltaT*( A2loc - A2)/(cumdeltaT);
    A3   =   A3  +  deltaT*( A3loc - A3)/(cumdeltaT);
    B   =   B  +  deltaT*( Bloc - B)/(cumdeltaT);
    C   =   C  +  deltaT*( Cloc - C)/(cumdeltaT);

    S1 = S1 + (A3loc - A3old)*(A3loc - A3);

    if (iter < gradIter)
      gradError[iter] = A3loc;

    iter++;
    
    if (iter == gradIter-1) {
      rk = calcTcorr(gradError);
    }

    //update the walker
    if (true) {
      int I = excitation1[nextDet]/2/norbs, A = excitation1[nextDet] - 2*norbs*I;
      if (I%2 == 0) walk.updateA(I/2, A/2, w);
      else walk.updateB(I/2, A/2, w);

      if (excitation2[nextDet] != 0) {
	int I = excitation2[nextDet]/2/norbs, A = excitation2[nextDet] - 2*norbs*I;

	if (I%2 == 0) walk.updateA(I/2, A/2, w);
	else walk.updateB(I/2, A/2, w);
      }
      ovlpRatio.clear(); excitation1.clear(); excitation2.clear();

      w.PTcontribution3rdOrder(walk, E0, I1, I2, I2hb, coreE, A2loc, Bloc, Cloc, A3loc,
			       ovlpRatio, excitation1, excitation2, false); 
    }
    
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &A2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &A3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  A2 = A2/commsize;
  A3 = A3/commsize;
  B = B/commsize;
  C = C/commsize;

  stddev = sqrt(S1*rk/(niter-1)/niter/commsize) ;
#ifndef SERIAL
  MPI_Bcast(&stddev    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

  if (commrank == 0) cout << A2<<"  "<<A3<<endl;
  return A2 + B*B/C;
}







