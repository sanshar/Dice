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
#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;
using namespace Eigen;

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
	d.setocc(2*alphaDets[a][i], true);
      for (int i=0; i<betaDets[b].size(); i++)
	d.setocc(2*betaDets[b][i]+1, true);
      allDets.push_back(d);
    }

  alphaDets.clear(); betaDets.clear();

  VectorXd dovlpPsi  =VectorXd::Zero(allDets.size());
  VectorXd dovlpPsi_t=VectorXd::Zero(allDets.size());

  for (int i=0; i<allDets.size(); i++) {
    dovlpPsi[i] = w.Overlap(allDets[i]);
  }

  //MatrixXd Ham = MatrixXd::Zero(allDets.size(), allDets.size());
  for (int i=commrank; i<allDets.size(); i+=commsize) {
    dovlpPsi_t[i] += (allDets[i].Energy(I1, I2, coreE)-E0)*dovlpPsi[i];
    //Ham(i,i) = allDets[i].Energy(I1, I2, coreE) - E0;
    for (int j=i+1; j<allDets.size(); j++) 
      if (allDets[i].connected(allDets[j])) {
	size_t orbDiff = 0;
	double Hamij = Hij(allDets[i], allDets[j], I1, I2, coreE, orbDiff);
	dovlpPsi_t[i] += Hamij*dovlpPsi[j];
	dovlpPsi_t[j] += Hamij*dovlpPsi[i];
      }
  }

  int size = allDets.size();
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(dovlpPsi_t[0]),     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif


  double ovlp = dovlpPsi.transpose()*dovlpPsi;


  for (int i=commrank; i<allDets.size(); i+=commsize) {
    double factor = dovlpPsi_t[i]/ovlp;
    w.OverlapWithGradient(allDets[i], 
			  factor, grad);
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]),     grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  //exit(0);
}


//<psi_t| (H0-E0)^-1 (H-E0) |psi>
void getGradientUsingDavidson(Wfn& w, double& E0, int& nalpha, int& nbeta, int& norbs,
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
	d.setocc(2*alphaDets[a][i], true);
      for (int i=0; i<betaDets[b].size(); i++)
	d.setocc(2*betaDets[b][i]+1, true);
      allDets.push_back(d);
    }

  alphaDets.clear(); betaDets.clear();

  VectorXd dovlpPsi  =VectorXd::Zero(allDets.size());
  VectorXd diag      =VectorXd::Zero(allDets.size());
  VectorXd dovlpPsi_t=VectorXd::Zero(allDets.size());

  for (int i=0; i<allDets.size(); i++) {
    dovlpPsi[i] = w.Overlap(allDets[i]);
  }

  //MatrixXd Ham = MatrixXd::Zero(allDets.size(), allDets.size());
  for (int i=commrank; i<allDets.size(); i+=commsize) {
    diag[i] = allDets[i].Energy(I1, I2, coreE);
    dovlpPsi_t[i] += (diag[i]-E0)*dovlpPsi[i];
    //Ham(i,i) = allDets[i].Energy(I1, I2, coreE) - E0;
    for (int j=i+1; j<allDets.size(); j++) 
      if (allDets[i].connected(allDets[j])) {
	size_t orbDiff = 0;
	double Hamij = Hij(allDets[i], allDets[j], I1, I2, coreE, orbDiff);
	dovlpPsi_t[i] += Hamij*dovlpPsi[j];
	dovlpPsi_t[j] += Hamij*dovlpPsi[i];
      }
  }

  int size = allDets.size();
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(dovlpPsi_t[0]),     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(diag[0])      ,     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  for (int i=0; i<allDets.size(); i++) 
    dovlpPsi_t[i] /= (E0-diag[i]);


  double ovlp = dovlpPsi.transpose()*dovlpPsi;


  for (int i=commrank; i<allDets.size(); i+=commsize) {
    double factor = dovlpPsi_t[i]/ovlp;
    w.OverlapWithGradient(allDets[i], 
			  factor, grad);
  }
#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(grad[0]),     grad.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  //exit(0);
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
	d.setocc(2*alphaDets[a][i], true);
      for (int i=0; i<betaDets[b].size(); i++)
	d.setocc(2*betaDets[b][i]+1, true);
      allDets.push_back(d);
    }

  alphaDets.clear(); betaDets.clear();

  double E, ovlp;
  for (int d=commrank; d<allDets.size(); d+=commsize) {
    double Eloc, ovlploc; 
    w.HamAndOvlp(allDets[d], ovlploc, Eloc, I1, I2, coreE);
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
