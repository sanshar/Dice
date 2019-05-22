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
#include "Correlator.h"
#include "Determinants.h"
#include "global.h"
#include <vector>
#include "input.h"

Correlator::Correlator (std::vector<int>& pasites,
                        std::vector<int>& pbsites,
                        double iv) : asites(pasites), bsites(pbsites) {
  if (asites.size()+bsites.size() > 20) {
    std::cout << "Cannot handle correlators of size greater than 20."<<std::endl;
    exit(0);
  }
  std::sort(asites.begin(), asites.end());
  std::sort(bsites.begin(), bsites.end());
  if(!schd.expCorrelator)
  {// exponential correlator coefficients 
    Variables.resize( pow(2,asites.size()+bsites.size()), iv);
  }
  else
  {
    Variables.resize( pow(2,asites.size()+bsites.size()), 0.0);
    //Variables.resize( pow(2,asites.size()+bsites.size()), 1.0);
  }
}

void Correlator::OverlapWithGradient(const Determinant& d, 
				     Eigen::VectorBlock<Eigen::VectorXd>& grad,
				     const double& ovlp,
				     const long& startIndex) const {
  
  int asize = asites.size();
  long index=0, one=1, index2=0;
  for (int n=0; n<asize; n++)
    if (d.getoccA( asites[n])) {
      index |= (one<< (n+asize));
    }

  for (int n=0; n<asize; n++)
    if (d.getoccB( bsites[n])) {
      index |= (one<< (n));
    }
  if(!schd.expCorrelator)
  {
      grad[index+startIndex] += ovlp/Variables[index];
  }
  else
  {
      grad[index+startIndex] += ovlp;
      //grad[index+startIndex] += 2*ovlp/Variables[index];
  }
}

double Correlator::Overlap(const BigDeterminant& d) const {

  double Coefficient = 0.0;
  int asize = asites.size();

  long index=0, one=1;
  for (int n=0; n<asize; n++)
    if (d [2*asites[n]] == 1)
      index |= (one << (n+asize));

  for (int n=0; n<asize; n++)
    if (d [2*bsites[n]+1] == 1)
      index |= (one<<(n));
  

  if(!schd.expCorrelator)
  {
      return Variables[index];
  }
  else
  {
      return exp(Variables[index]);
      //return (Variables[index]*Variables[index]);
  }
}

double Correlator::Overlap(const Determinant& d) const {

  double Coefficient = 0.0;
  int asize = asites.size();

  long index=0, one=1;
  for (int n=0; n<asize; n++)
    if (d.getoccA( asites[n]))
      index |= (one << (n+asize));

  for (int n=0; n<asize; n++)
    if (d.getoccB( bsites[n]))
      index |= (one<< (n));
  
  if(!schd.expCorrelator)
  {
      return Variables[index];
  }
  else
  {
      return exp(Variables[index]);
      //return (Variables[index]*Variables[index]);
  }
}

double Correlator::OverlapRatio(const Determinant& d1, const Determinant& d2) const {

  double Coefficient = 0.0;
  int asize = asites.size();
  
  long index1=0, index2=0, one=1;
  for (int n=0; n<asize; n++) {
    if (d1.getoccA( asites[n]))
      index1 |= (one<< (n+asize));
    if (d2.getoccA( asites[n]))
      index2 |= (one<< (n+asize));
  }
  for (int n=0; n<asize; n++) {
    if (d1.getoccB( bsites[n]))
      index1 |= (one<< (n));
    if (d2.getoccB( bsites[n]))
      index2 |= (one<< (n));
  }
  
  if(!schd.expCorrelator)
  {
      return Variables[index1]/Variables[index2];
  }
  else
  {
    return exp(Variables[index1]-Variables[index2]);
  }
}


double Correlator::OverlapRatio(const BigDeterminant& d1, const BigDeterminant& d2) const {

  double Coefficient = 0.0;
  int asize = asites.size();

  long index1=0, index2=0, one=1;
  for (int n=0; n<asize; n++) {
    if (d1[2*asites[n]] == 1)
      index1 |= (one<< (n+asize));
    if (d2[2*asites[n]] == 1)
      index2 |= (one<< (n+asize));
  }
  for (int n=0; n<asize; n++) {
    if (d1[2*bsites[n]+1] == 1)
      index1 |= (one<< (n));
    if (d2[2*bsites[n]+1] == 1)
      index2 |= (one<< (n));
  }
  
  if(!schd.expCorrelator)
  {
      return Variables[index1]/Variables[index2];
  }
  else
  {
    return exp(Variables[index1]-Variables[index2]);
  }
}

std::ostream& operator<<(std::ostream& os, const Correlator& c) {
  for (int i=0; i<c.asites.size(); i++)
    os << c.asites[i]<<"a  ";
  for (int i=0; i<c.bsites.size(); i++)
    os << c.bsites[i]<<"b  ";
  os<<std::endl;
  return os;
}
