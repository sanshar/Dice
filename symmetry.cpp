/*
   Developed by Sandeep Sharma with contributions from James E. Smith and Adam A. Homes, 2017
   Copyright (c) 2017, Sandeep Sharma

   This file is part of DICE.
   This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "global.h"
#include <Eigen/Dense>
#include "Determinants.h"
#include <iostream>
#include <string>

using namespace Eigen;
using namespace std;

class symmetry
{
// Product tables for Abelian point groups

/* D2h (Using MolPro numerical notation)
   Ag   1
   B3u  2
   B2u  3
   B1g  4
   B1u  5
   B2g  6
   B3g  7
   Au   8
 */
static MatrixXd d2h(8,8);
d2h(0,0) = 1; d2h(1,0) = 2; d2h(2,0) = 3; d2h(3,0) = 4;
d2h(0,1) = 2; d2h(1,1) = 1; d2h(2,1) = 4; d2h(3,1) = 3;
d2h(0,2) = 3; d2h(1,2) = 4; d2h(2,2) = 1; d2h(3,2) = 2;
d2h(0,3) = 4; d2h(1,3) = 3; d2h(2,3) = 2; d2h(3,3) = 1;
d2h(0,4) = 5; d2h(1,4) = 6; d2h(2,4) = 6; d2h(3,4) = 8;
d2h(0,5) = 6; d2h(1,5) = 5; d2h(2,5) = 5; d2h(3,5) = 7;
d2h(0,6) = 7; d2h(1,6) = 8; d2h(2,6) = 8; d2h(3,6) = 6;
d2h(0,7) = 8; d2h(1,7) = 7; d2h(2,7) = 7; d2h(3,7) = 5;

d2h(4,0) = 5; d2h(5,0) = 6; d2h(6,0) = 7; d2h(7,0) = 8;
d2h(4,1) = 6; d2h(5,1) = 5; d2h(6,1) = 8; d2h(7,1) = 7;
d2h(4,2) = 7; d2h(5,2) = 8; d2h(6,2) = 5; d2h(7,2) = 6;
d2h(4,3) = 8; d2h(5,3) = 7; d2h(6,3) = 6; d2h(7,3) = 5;
d2h(4,4) = 1; d2h(5,4) = 2; d2h(6,4) = 3; d2h(7,4) = 4;
d2h(4,5) = 2; d2h(5,5) = 1; d2h(6,5) = 4; d2h(7,5) = 3;
d2h(4,6) = 3; d2h(5,6) = 4; d2h(6,6) = 1; d2h(7,6) = 2;
d2h(4,7) = 4; d2h(5,7) = 3; d2h(6,7) = 2; d2h(7,7) = 1;

// Functions

// Here i and j should be the MolPro notation for irreps
static int product(int i, int j)
{
	return d2h(i-1,j-1);
}

}

int main()
{
	string pg = "Ag";
	cout << d2h << endl;
	cout << d2h(2,3) << endl;

	return 0;

}
