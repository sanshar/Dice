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
#include <algorithm>
#include "integral.h"
#include "Determinants.h"
#include <boost/format.hpp>
#include <iostream>
#include <fstream>
#include "evaluateE.h"
#include <math.h>
#include "global.h"
#include "input.h"
#include "statistics.h"
#include "sr.h"

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;
using namespace Eigen;

/*
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
  //vector<double> w(v.size(), 1);
  int n = v.size();
  double norm, rk, f, neff;

  double aver = 0, var = 0;
  for (int i = 0; i < v.size(); i++)
  {
    aver += v[i] ;
    norm += 1.0 ;
  }
  aver = aver / norm;

  neff = 0.0;
  for (int i = 0; i < n; i++)
  {
    neff = neff + 1.0;
  };
  neff = norm * norm / neff;

  for (int i = 0; i < v.size(); i++)
  {
    var = var + (v[i] - aver) * (v[i] - aver);
  };
  var = var / norm;
  var = var * neff / (neff - 1.0);

  //double c[v.size()];
  vector<double> c(v.size(),0);
  //for (int i=0; i<v.size(); i++) c[i] = 0.0;
  int l = v.size() - 1;

  int i = commrank+1;
  for (; i < l; i+=commsize)
  //int i = 1;
  //for (; i < l; i++)
  {
    c[i] = 0.0;
    double norm = 0.0;
    for (int k = 0; k < n - i; k++)
    {
      c[i] = c[i] + (v[k] - aver) * (v[k + i] - aver);
      norm = norm + 1.0;
    };
    c[i] = c[i] / norm / var;
  };
 #ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &c[0], v.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  rk = 1.0;

  f = 1.0;

  i = 1;
  ofstream out("OldCorrFunc.txt");
  for (; i < l; i++)
  {

    if (commrank == 0)
    {
      out << c[i] << endl;
    }

    if (c[i] < 0.0)
      f = 0.0;
    rk = rk + 2.0 * c[i] * f;
  }
  out.close();
  
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
*/
