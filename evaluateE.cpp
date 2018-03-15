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

  MatrixXd Ham = MatrixXd::Zero(allDets.size(), allDets.size());
  for (int i=0; i<allDets.size(); i++) {
    Ham(i,i) = allDets[i].Energy(I1, I2, coreE) - E0;
    for (int j=i+1; j<allDets.size(); j++) 
      if (allDets[i].connected(allDets[j])) {
	size_t orbDiff = 0;
	Ham(i,j) = Hij(allDets[i], allDets[j], I1, I2, coreE, orbDiff);
	Ham(j,i) = Ham(i,j);
      }
  }


  VectorXd dovlpPsi  =VectorXd::Zero(allDets.size());
  VectorXd dovlpPsi_t=VectorXd::Zero(allDets.size());

  for (int i=0; i<allDets.size(); i++) {
    dovlpPsi[i] = w.Overlap(allDets[i]);
  }

  double ovlp = dovlpPsi.transpose()*dovlpPsi;

  dovlpPsi_t = Ham*dovlpPsi;

  for (int i=0; i<allDets.size(); i++) {
    double factor = dovlpPsi_t[i]/ovlp;
    w.OverlapWithGradient(allDets[i], 
			  factor, grad);
  }

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

  double E, ovlp;
  for (int d=0; d<allDets.size(); d++) {
    double Eloc, ovlploc; 
    w.HamAndOvlp(allDets[d], ovlploc, Eloc, I1, I2, coreE);
    E += ovlploc*Eloc;
    ovlp += ovlploc*ovlploc;
    //cout << Eloc<<"  "<<ovlploc<<"  "<<E<<"  "<<ovlp<<"  "<<E/ovlp<<endl;
  }
  //cout << E<<"  "<<ovlp<<"  "<<E/ovlp<<endl;
  return E/ovlp;
}
