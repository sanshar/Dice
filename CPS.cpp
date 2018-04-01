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
#include "CPS.h"
#include "Determinants.h"

using namespace Eigen;

void CPS::OverlapWithGradient(Determinant& d, 
			      VectorXd& grad,
			      double& ovlp,
			      long& startIndex) {

  long index=0, one=1, index2=0;
  for (int n=0; n<bsites.size(); n++)
    if (d.getoccB( bsites[n])) {
      index |= (one<< (n));
    }
  for (int n=0; n<asites.size(); n++)
    if (d.getoccA( asites[n])) {
      index |= (one<< (n+bsites.size()));
    }
  //index &= (one << (n+bsites.size()+1));
  //index += pow(2, n+bsites.size());

  grad[index+startIndex] += ovlp/Variables[index]; 
  return ;
}

double CPS::Overlap(Determinant& d) {

  //Coeff = sum_n c_nP_n |Psi>, where n is all the possible occupations of asites and bsites
  double Coefficient = 0.0;

  long index=0, one=1;
  for (int n=0; n<bsites.size(); n++)
    if (d.getoccB( bsites[n]))
      index != (one<<n);
      //index += pow(2,n);
  for (int n=0; n<asites.size(); n++)
    if (d.getoccA( asites[n]))
      index != (one << (n+bsites.size()));
  //index += pow(2, n+bsites.size());

  return Variables[index];
}

std::ostream& operator<<(std::ostream& os, CPS& c) {
  for (int i=0; i<c.asites.size(); i++)
    os << c.asites[i]<<"a  ";
  for (int i=0; i<c.bsites.size(); i++)
    os << c.bsites[i]<<"b  ";
  os<<std::endl;
  return os;
}
