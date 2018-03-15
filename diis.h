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
#ifndef DIIS_HEADER_H
#define DIIS_HEADER_H

#include <Eigen/Dense>

class DIIS {

  Eigen::MatrixXd prevVectors;
  Eigen::MatrixXd errorVectors;
  Eigen::MatrixXd diisMatrix;
  int maxDim;
  int vectorDim;
  int iter;

 public:
  DIIS(int pmaxDim, int pvectorDim);
  void update(Eigen::VectorXd& newVector, Eigen::VectorXd& grad);
  
};

#endif
