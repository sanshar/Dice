/* Copyright (c) 2012  Gerald Knizia
 * 
 * This file is part of the IR/WMME program
 * (See https://sites.psu.edu/knizia/)
 * 
 * IR/WMME is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 * 
 * IR/WMME is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with bfint (LICENSE). If not, see http://www.gnu.org/licenses/
 */

#pragma once
#include <vector>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

struct BasisShell
{
  double Xcoord, Ycoord, Zcoord; //center of the function
  int l; //angular momentum
  int nFn, nCo; //number of contracted and primitive functions
  vector<double> exponents;
  MatrixXd contractions; //nFn x nCo
  
  
  double fCo(unsigned iExp, unsigned iCo) { return contractions(iExp, iCo); }
  double fExp(unsigned iExp) { return exponents[iExp]; }
  void PrintAligned(std::ostream &xout, uint Indent) const;
  size_t numFuns() const {return nCo * (2 * l + 1);}
  size_t nSh() const {return (2 * l + 1);}
};

struct BasisSet
{
  vector<BasisShell> BasisShells;
  void PrintAligned(std::ostream &xout, uint Indent) const {
    for (int i=0; i<BasisShells.size(); i++) {
      xout << endl;
      BasisShells[i].PrintAligned(xout, Indent);
    }
    xout << endl;
  }
  int getNbas();
  int getNbas(int shlIndex);
  int getNPrimitivebas(int shlIndex);
};

double RawGaussNorm(double fExp, unsigned l);
