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

void Correlator::OverlapWithGradient(const Determinant& d, 
				     VectorXd& grad,
				     const double& ovlp,
				     const long& startIndex) {
  
  long index=0, one=1, index2=0;
  for (int n=0; n<bsites.size(); n++)
    if (d.getoccB( bsites[n])) {
      index |= (one<< (n));
    }
  for (int n=0; n<asites.size(); n++)
    if (d.getoccA( asites[n])) {
      index |= (one<< (n+bsites.size()));
    }

  grad[index+startIndex] += ovlp/Variables[index]; 
  return ;
}

double Correlator::Overlap(const Determinant& d) {

  double Coefficient = 0.0;

  long index=0, one=1;
  for (int n=0; n<bsites.size(); n++)
    if (d.getoccB( bsites[n]))
      index |= (one<<n);
  
  for (int n=0; n<asites.size(); n++)
    if (d.getoccA( asites[n]))
      index |= (one << (n+bsites.size()));

  return Variables[index];
}

double Correlator::OverlapRatio(const Determinant& d1, const Determinant& d2) {

  double Coefficient = 0.0;

  long index1=0, index2=0, one=1;
  for (int n=0; n<bsites.size(); n++) {
    if (d1.getoccB( bsites[n]))
      index1 |= (one<<n);
    if (d2.getoccB( bsites[n]))
      index2 |= (one<<n);
  }
  
  for (int n=0; n<asites.size(); n++) {
    if (d1.getoccA( asites[n]))
      index1 |= (one<< (n+bsites.size()));
    if (d2.getoccA( asites[n]))
      index2 |= (one<< (n+bsites.size()));
  }

  return Variables[index1]/Variables[index2];
}

std::ostream& operator<<(std::ostream& os, const Correlator& c) {
  for (int i=0; i<c.asites.size(); i++)
    os << c.asites[i]<<"a  ";
  for (int i=0; i<c.bsites.size(); i++)
    os << c.bsites[i]<<"b  ";
  os<<std::endl;
  return os;
}
