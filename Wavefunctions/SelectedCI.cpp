/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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
#include "Determinants.h"
#include "SimpleWalker.h"
#include "SelectedCI.h"

SelectedCI::SelectedCI() {readWave();}

void SelectedCI::readWave() {
  int norbs = Determinant::norbs;
  int nalpha = Determinant::nalpha;
  int nbeta = Determinant::nbeta;
  if (boost::iequals(schd.determinantFile, ""))
  {
    Determinant det;

    for (int i = 0; i < nalpha; i++)
      det.setoccA(i, true);
    for (int i = 0; i < nbeta; i++)
      det.setoccB(i, true);
    DetsMap[det] = 0;
    coeffs.push_back(1.0);
    bestDeterminant = det;
      
  }
  else
  {
    ifstream dump(schd.determinantFile.c_str());
    int index = 0;
    double bestCoeff = 0.0;
    while (dump.good())
    {
      std::string Line;
      std::getline(dump, Line);

      trim_if(Line, is_any_of(", \t\n"));
      
      vector<string> tok;
      boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);

      if (tok.size() > 2 )
      {
        double ci = atof(tok[0].c_str());
        Determinant det ;
        for (int i=0; i<Determinant::norbs; i++) 
        {
          if (boost::iequals(tok[1+i], "2")) 
          {
            det.setoccA(i, true);
            det.setoccB(i, true);
          }
          else if (boost::iequals(tok[1+i], "a")) 
          {
            det.setoccA(i, true);
            det.setoccB(i, false);
          }
          if (boost::iequals(tok[1+i], "b")) 
          {
            det.setoccA(i, false);
            det.setoccB(i, true);
          }
          if (boost::iequals(tok[1+i], "0")) 
          {
            det.setoccA(i, false);
            det.setoccB(i, false);
          }
        }

        DetsMap[det] = index;
        coeffs.push_back(ci);
        if (abs(ci) > abs(bestCoeff)) {
          bestCoeff = ci;
          bestDeterminant = det;
        }
      }
    }

  }
}

double SelectedCI::getOverlapFactor(SimpleWalker& walk, Determinant& dcopy) {
  auto it1 = DetsMap.find(walk.d);
  auto it2 = DetsMap.find(dcopy);
  if (it1 != DetsMap.end() && it2 != DetsMap.end())
    return coeffs[it2->second]/coeffs[it1->second];
  else
    return 0.0;
}

double SelectedCI::getOverlapFactor(int I, int A, SimpleWalker& walk, bool doparity) {
  Determinant dcopy = walk.d;
  dcopy.setocc(I, false);
  dcopy.setocc(A, true);
  auto it1 = DetsMap.find(walk.d);
  auto it2 = DetsMap.find(dcopy);
  if (it1 != DetsMap.end() && it2 != DetsMap.end())
    return coeffs[it2->second]/coeffs[it1->second];
  else
    return 0.0;
}

double SelectedCI::getOverlapFactor(int I, int J, int A, int B,
                                    SimpleWalker& walk, bool doparity) {
  Determinant dcopy = walk.d;
  dcopy.setocc(I, false);
  dcopy.setocc(J, false);
  dcopy.setocc(A, true);
  dcopy.setocc(B, true);
  auto it1 = DetsMap.find(walk.d);
  auto it2 = DetsMap.find(dcopy);
  if (it1 != DetsMap.end() && it2 != DetsMap.end())
    return coeffs[it2->second]/coeffs[it1->second];
  else
    return 0.0;
}
  
double SelectedCI::Overlap(SimpleWalker& walk) {
  auto it1 = DetsMap.find(walk.d);
  if (it1 != DetsMap.end())
    return coeffs[it1->second];
  else
    return 0.0;
}

void SelectedCI::OverlapWithGradient(SimpleWalker &walk,
                                     double &factor,
                                     Eigen::VectorXd &grad) {
  auto it1 = DetsMap.find(walk.d);
  if (it1 != DetsMap.end())
    grad[it1->second] = 1.0;
}

void SelectedCI::getVariables(Eigen::VectorXd &v) {
  for (int i=0; i<v.rows(); i++)
    v[i] = coeffs[i];
}

long SelectedCI::getNumVariables() {
  return coeffs.size();
}

void SelectedCI::updateVariables(Eigen::VectorXd &v) {
  for (int i=0; i<v.rows(); i++)
    coeffs[i] = v[i];
}

void SelectedCI::printVariables() {
  for (int i=0; i<coeffs.size(); i++)
    cout << coeffs[i]<<endl;
}

