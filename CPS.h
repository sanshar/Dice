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
#ifndef CPS_HEADER_H
#define CPS_HEADER_H
#include <Eigen/Dense>
#include <vector>
#include <iostream>

class Determinant;

//A correlator for a bunch of spin sites
//e.g. for a 2 site correlator Cab
//you have the following variables
// C00, C10, C01, C11 , 

//Similarty for n site correlator one can have 2^n variables
class CPS {
 public:
  //b+1 b+2 ...         0 1 2..
  std::vector<int> asites, bsites;

  std::vector<double> Variables; //2^{na+nb} number of variables

 CPS(std::vector<int>& pasites, std::vector<int>& pbsites, double iv=1.0) : asites(pasites), bsites(pbsites) {
    if (asites.size()+bsites.size() > 5) {
      std::cout << "Cannot handle correlators of size greater than 5."<<std::endl;
      exit(0);
    }
    Variables.resize( pow(2,asites.size()+bsites.size()), iv);
  }

  double Overlap            (Determinant& d);
  void   OverlapWithGradient(Determinant& d, 
			     Eigen::VectorXd& grad,
			     double& ovlp,
			     long& startIndex);

  friend std::ostream& operator<<(std::ostream& os, CPS& c); 
};


#endif
