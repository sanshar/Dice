/*
   Developed by Sandeep Sharma with contributions from James E. Smith
   and Adam A. Homes, 2017
   Copyright (c) 2017, Sandeep Sharma

   This file is part of DICE.
   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
   details.

   You should have received a copy of the GNU General Public License along with
   this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef symmetry_HEADER_H
#define symmetry_HEADER_H

#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include <algorithm>
#include <boost/bind.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "global.h"

using namespace Eigen;
using namespace std;

class oneInt;
class Determinant;

bool compareForSortingEnergies(const pair<double, int>& a,
                               const pair<double, int>& b);

class symmetry {
 private:
  // Full table for d2h and all subgroups
  MatrixXd fullD2hTable;

 public:
  bool init_success;
  static MatrixXd product_table;   // product table for a given symmetry
  string pointGroup;               // eg. d2h, c2
  symmetry(string, vector<int>&);  // product and sym is initialized
  int getProduct(int, int);
  int getProduct(vector<int>&);
  int getSymmetry(char*, vector<int>&);
  void estimateLowestEnergyDet(int, int, oneInt, vector<int>&, vector<int>&,
                               Determinant&);
};

#endif
