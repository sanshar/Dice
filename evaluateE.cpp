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

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;
using namespace Eigen;
double calcTcorr(vector<double>& v) {
  vector<double> w(v.size(), 1);
  int n = w.size();
  double norm, rk, f, neff;

  double aver=0, var=0;
  for (int i=0; i<w.size(); i++) {
    aver += v[i]*w[i];
    norm += w[i];
  }
  aver = aver/norm;

  neff = 0.0;
  for(int i=0;i<n;i++){
    neff = neff+w[i]*w[i];
  };
  neff = norm*norm/neff;

  for(int i=0;i<v.size();i++){
    var = var+w[i]*(v[i]-aver)*(v[i]-aver);
  };
  var = var/norm;
  var = var*neff/(neff-1.0);
  
  double c[1000];
  int l = w.size()-1;
  for(int i=1;i<l;i++){
    c[i] = 0.0;
    norm = 0.0;
    for(int k=0;k<n-i;k++){
      c[i] = c[i] + sqrt(w[k]*w[k+i])*(v[k]-aver)*(v[k+i]-aver);
      norm = norm + sqrt(w[k]*w[k+i]);
    };
    c[i] = c[i]/norm/var;
  };
  rk = 1.0;
  f  = 1.0;
  for(int i=1;i<l;i++){
    if(c[i]<0.0) f=0.0;
    rk = rk+2.0*c[i]*f;
  };

  return rk;
}


void getStochasticGradientUsingDavidson(CPSSlater& w, double& E0, int& nalpha, int& nbeta, int& norbs,
					oneInt& I1, twoInt& I2, double& coreE, 
					VectorXd& grad, int niter, double targetError) 
{
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  generator);

  //initialize the walker
  Determinant d;
  for (int i=0; i<nalpha; i++)
    d.setoccA(i, true);
  for (int j=0; j<nbeta; j++)
    d.setoccB(j, true);
  Walker walk(d);
  walk.initUsingWave(w);
  
  int iter = 0;
  double cumulative = 0., cumulative2 = 0., Eavg=0., stddev=1.e4;
  double rk, Eloc=0.;
  double ovlp= 0., ham=0.;
  VectorXd localGrad = grad; localGrad.setZero();
  double scale = 1.0;
 
  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, scale, E0, I1, I2, coreE); 
  
  std::vector<double> gradError(1000, 0);
  while (iter <niter && stddev >targetError) {
    grad += localGrad/ovlp;
    Eloc += ham/ovlp;

    double locgradError = localGrad.squaredNorm();
    if (iter <1000) gradError[iter] = locgradError;
    
    cumulative  += locgradError;
    cumulative2 += locgradError*locgradError;
    iter++;
    
    if (iter %1000 == 0) {
      if (iter == 1000 && commrank == 0) rk = calcTcorr(gradError);
      double cum = cumulative, cum2 = cumulative2;
      int cumiter = iter*commsize;
#ifndef SERIAL
      MPI_Allreduce(&cumulative,  &cum,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&cumulative2, &cum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      if (commrank == 0) {
	stddev = pow(  (cum2*cumiter - cum*cum)*rk/cumiter/(cumiter-1)/cumiter, 0.5);
	Eavg  = cum/cumiter;
	//std::cout << boost::format("%6i   %14.8f  %14.8f %14.8f\n") %iter % Eavg % stddev % rk;
      }
#ifndef SERIAL
      MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&rk    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }
    
    
    //pick a random occupied orbital
    int i = floor( random()*(nalpha+nbeta) );
    if (i < nalpha) {
      int a = floor(random()* (norbs-nalpha) );
      double detfactor = walk.getDetFactorA(i, a, w);
      if ( pow(detfactor, 2) > random() ) {
	walk.updateA(i, a, w);
	localGrad.setZero();
	w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, scale, E0, I1, I2, coreE); 
      }
      
    }
    else {
      i = i - nalpha;
      int a = floor( random()*(norbs-nbeta));
      double detfactor = walk.getDetFactorB(i, a, w);
      
      if ( pow(detfactor, 2) > random() ) {
	walk.updateB(i, a, w);
	localGrad.setZero();
	w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, scale, E0, I1, I2, coreE); 
      }
      
    }
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]),     grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  grad /= (iter*commsize);
  E0 = Eloc/(iter*commsize);
}


void getStochasticGradient(CPSSlater& w, double& E0, int& nalpha, int& nbeta, int& norbs,
			   oneInt& I1, twoInt& I2, double& coreE, 
			   VectorXd& grad, int niter, double targetError) 
{
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  generator);

  //initialize the walker
  Determinant d;
  for (int i=0; i<nalpha; i++)
    d.setoccA(i, true);
  for (int j=0; j<nbeta; j++)
    d.setoccB(j, true);
  Walker walk(d);
  walk.initUsingWave(w);
  
  int iter = 0;
  double cumulative = 0., cumulative2 = 0., Eavg=0., stddev=1.e4;
  double rk, Eloc=0.;
  double ham=0., ovlp =0.;
  VectorXd localGrad = grad; localGrad.setZero();
  double scale = 1.0;
 
  w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, scale, E0, I1, I2, coreE); 


  std::vector<double> gradError(1000, 0);
  while (iter <niter && stddev >targetError) {
    grad += localGrad/ovlp;
    Eloc += ham/ovlp;

    double locgradError = localGrad.squaredNorm();
    if (iter <1000) gradError[iter] = locgradError;
    
    cumulative  += locgradError;
    cumulative2 += locgradError*locgradError;
    iter++;
    
    if (iter %1000 == 0) {
      if (iter == 1000 && commrank == 0) rk = calcTcorr(gradError);
      double cum = cumulative, cum2 = cumulative2;
      int cumiter = iter*commsize;
#ifndef SERIAL
      MPI_Allreduce(&cumulative,  &cum,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&cumulative2, &cum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      if (commrank == 0) {
	stddev = pow(  (cum2*cumiter - cum*cum)*rk/cumiter/(cumiter-1)/cumiter, 0.5);
	Eavg  = cum/cumiter;
	//std::cout << boost::format("%6i   %14.8f  %14.8f %14.8f\n") %iter % Eavg % stddev % rk;
      }
#ifndef SERIAL
      MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&rk    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }
    
    
    //pick a random occupied orbital
    int i = floor( random()*(nalpha+nbeta) );
    if (i < nalpha) {
      int a = floor(random()* (norbs-nalpha) );
      double detfactor = walk.getDetFactorA(i, a, w);
      if ( pow(detfactor, 2) > random() ) {
	walk.updateA(i, a, w);
	localGrad.setZero();
	w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, scale, E0, I1, I2, coreE); 
      }
      
    }
    else {
      i = i - nalpha;
      int a = floor( random()*(norbs-nbeta));
      double detfactor = walk.getDetFactorB(i, a, w);
      
      if ( pow(detfactor, 2) > random() ) {
	walk.updateB(i, a, w);
	localGrad.setZero();
	w.HamAndOvlpGradient(walk, ovlp, ham, localGrad, scale, E0, I1, I2, coreE); 
      }
      
    }
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]),     grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &Eloc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  grad /= (iter*commsize);
  E0 = Eloc/(iter*commsize);
  
}

void comb(int N, int K, vector<vector<int> >& combinations)
{
  std::vector<int> bitmask(K,1);
  bitmask.resize(N, 0); // N-K trailing 0's

  // print integers and permute bitmask
  int index = 0;
  do {
    vector<int> comb;
    for (int i = 0; i < N; ++i) // [0..N-1] integers
      {
	if (bitmask[i]==1) comb.push_back(i);
      }
    combinations.push_back(comb);
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
}


//<psi_t| (H-E0) |psi>
void getGradient(Wfn& w, double& E0, int& nalpha, int& nbeta, int& norbs,
		 oneInt& I1, twoInt& I2, double& coreE,
		 VectorXd& grad) {
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

  double Overlap = 0;
  grad.setZero();
  for (int i=commrank; i<allDets.size(); i+=commsize) {
    double overlap = w.Overlap(allDets[i]);
    Walker walk(allDets[i]);walk.initUsingWave(w);
    double ovlp, ham, Epsi;
    w.HamAndOvlpGradient(walk, ovlp, ham, grad, overlap, E0, I1, I2, coreE);
    Overlap += ovlp*ovlp;
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]),     grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap),               1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  grad /= Overlap;
}


//<psi_t| (H0-E0)^-1 (H-E0) |psi>
void getGradientUsingDavidson(CPSSlater& w, double& E0, int& nalpha, int& nbeta, int& norbs,
			      oneInt& I1, twoInt& I2, double& coreE,
			      VectorXd& grad) {

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
  double Overlap=0.;
  for (int i=commrank; i<allDets.size(); i+=commsize) {
    double overlap = w.Overlap(allDets[i]);
    Walker walk(allDets[i]);walk.initUsingWave(w);
    VectorXd gradlocal = grad; gradlocal.setZero();
    double ovlp, ham, Epsi;
    w.HamAndOvlpGradient(walk, ovlp, ham, grad, overlap, E0, I1, I2, coreE);
    Overlap += ovlp*ovlp;
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]),     grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(Overlap),               1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  grad /=Overlap;

}

//<psi|H|psi>/<psi|psi> = <psi|d> <d|H|psi>/<psi|d><d|psi>
double evaluateEDeterministic(Wfn& w, int& nalpha, int& nbeta, int& norbs,
			      oneInt& I1, twoInt& I2, double& coreE) {

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
  
  double E, ovlp;
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    double Eloc, ovlploc; 
    Walker walk(allDets[d]);
    walk.initUsingWave(w);
    w.HamAndOvlp(walk, ovlploc, Eloc, I1, I2, coreE);
    E += ovlploc*Eloc;
    ovlp += ovlploc*ovlploc;
  }
  allDets.clear();

  double Ebkp=E, obkp = ovlp;
  int size = 1;
#ifndef SERIAL
  MPI_Allreduce(&Ebkp, &E,     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&obkp, &ovlp,  size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  
  return E/ovlp;
}


//<psi|H|psi>/<psi|psi> = <psi|d> <d|H|psi>/<psi|d><d|psi>
double evaluateEStochastic(CPSSlater& w, int& nalpha, int& nbeta, int& norbs,
			   oneInt& I1, twoInt& I2, double& coreE,
			   int niter, double targetError) {

  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  generator);

  //initialize the walker
  Determinant d;
  for (int i=0; i<nalpha; i++)
    d.setoccA(i, true);
  for (int j=0; j<nbeta; j++)
    d.setoccB(j, true);
  Walker walk(d);
  walk.initUsingWave(w);


  int iter = 0;
  double cumulative = 0., cumulative2 = 0., Eavg=0., stddev=1.e4;
  double rk;
  double ovlp= 0., ham=0.;
  w.HamAndOvlp(walk, ovlp, ham, I1, I2, coreE);

  std::vector<double> Elocvec(1000, 0);

  while (iter < niter && stddev > targetError) {
    double Eloc = ham/ovlp;

    if (iter < 1000) Elocvec[iter] = Eloc;

    cumulative += Eloc;
    cumulative2 += Eloc*Eloc;
    iter ++;

    if (iter %1000 == 0) {
      if (iter == 1000 && commrank == 0) rk = calcTcorr(Elocvec);
      double cum = cumulative, cum2 = cumulative2;
      int cumiter = iter*commsize;
#ifndef SERIAL
      MPI_Allreduce(&cumulative,  &cum,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&cumulative2, &cum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      if (commrank == 0) {
	stddev = pow(  (cum2*cumiter - cum*cum)*rk/cumiter/(cumiter-1)/cumiter, 0.5);
	Eavg  = cum/cumiter;
	double Eavg2 = cum2/cumiter;
	std::cout << boost::format("%6i   %14.8f  %14.8f %14.8f\n") %iter % Eavg % stddev % rk;
      }
#ifndef SERIAL
      MPI_Bcast(&stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&rk    , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    }
    
    //pick a random occupied orbital
    int i = floor( random()*(nalpha+nbeta) );
    if (i < nalpha) {
      int a = floor(random()* (norbs-nalpha) );
      double detfactor = walk.getDetFactorA(i, a, w);
      if ( pow(detfactor, 2) > random() ) {
	walk.updateA(i, a, w);
	w.HamAndOvlp(walk, ovlp, ham, I1, I2, coreE);
      }

    }
    else {
      i = i - nalpha;
      int a = floor( random()*(norbs-nbeta));
      double detfactor = walk.getDetFactorB(i, a, w);

      if ( pow(detfactor, 2) > random() ) {
	walk.updateB(i, a, w);
	w.HamAndOvlp(walk, ovlp, ham, I1, I2, coreE);
      }
      
    }
    
  }
#ifndef SERIAL
  MPI_Bcast(&Eavg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  return Eavg;
}


