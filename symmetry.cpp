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
#include <vector>
#include <stdio.h>
#include <stdlib.h> 

using namespace Eigen;
using namespace std;

class symmetry
{
	// Product tables for Abelian point groups
	public:
    MatrixXd product_table; //product table for a given symmetry
    string pointGroup; //eg. d2h, c2
    symmetry(string); //product and sym is initialized
    int getProduct(int,int);
    int getSymmetry(char* repArray, vector<int>& irrep);
};

symmetry::symmetry( string pg )
{

  // Assign correct product table for point group. Note that even though the
  // indices of the columns are one less than the MolPro notation the irrep
  // returned is again in the MolPro notation.
  // TODO Should I delete the other tables since there are all just subtables of this?

  if ( pg == (string)"d2h" )
  {
    // D2h Ag:1, B1g:4, B2g:6, B3g:7, Au:8, B1u:5, B2u:3, B3u: 2 in MolPro not.
    // Row & col indices: Ag:0, B1g:3, B2g:5, B3g:6, Au:7, B1u:4, B2u:2, B3u:1
    pointGroup = pg;
    product_table.resize(8,8);
    product_table(0,0) = 1; product_table(1,0) = 2; product_table(2,0) = 3; product_table(3,0) = 4;
    product_table(0,1) = 2; product_table(1,1) = 1; product_table(2,1) = 4; product_table(3,1) = 3;
    product_table(0,2) = 3; product_table(1,2) = 4; product_table(2,2) = 1; product_table(3,2) = 2;
    product_table(0,3) = 4; product_table(1,3) = 3; product_table(2,3) = 2; product_table(3,3) = 1;
    product_table(0,4) = 5; product_table(1,4) = 6; product_table(2,4) = 6; product_table(3,4) = 8;
    product_table(0,5) = 6; product_table(1,5) = 5; product_table(2,5) = 5; product_table(3,5) = 7;
    product_table(0,6) = 7; product_table(1,6) = 8; product_table(2,6) = 8; product_table(3,6) = 6;
    product_table(0,7) = 8; product_table(1,7) = 7; product_table(2,7) = 7; product_table(3,7) = 5;

    product_table(4,0) = 5; product_table(5,0) = 6; product_table(6,0) = 7; product_table(7,0) = 8;
    product_table(4,1) = 6; product_table(5,1) = 5; product_table(6,1) = 8; product_table(7,1) = 7;
    product_table(4,2) = 7; product_table(5,2) = 8; product_table(6,2) = 5; product_table(7,2) = 6;
    product_table(4,3) = 8; product_table(5,3) = 7; product_table(6,3) = 6; product_table(7,3) = 5;
    product_table(4,4) = 1; product_table(5,4) = 2; product_table(6,4) = 3; product_table(7,4) = 4;
    product_table(4,5) = 2; product_table(5,5) = 1; product_table(6,5) = 4; product_table(7,5) = 3;
    product_table(4,6) = 3; product_table(5,6) = 4; product_table(6,6) = 1; product_table(7,6) = 2;
    product_table(4,7) = 4; product_table(5,7) = 3; product_table(6,7) = 2; product_table(7,7) = 1;
  }

  else if ( pg == (string)"c2v" )
  {
      //C2v A1: 1, B1:2, B2:3, A2:4 in MolPro. Indices here: A1: 0, B1:1, B2:2, A2:3
    pointGroup = pg;
    product_table.resize(4,4);
    product_table(0,0)= 1;  product_table(1,0)= 2;  product_table(2,0)= 3;  product_table(3,0)= 4;
    product_table(0,1)= 2;  product_table(1,1)= 1;  product_table(2,1)= 4;  product_table(3,1)= 3;
    product_table(0,2)= 3;  product_table(1,2)= 4;  product_table(2,2)= 1;  product_table(3,2)= 2;
    product_table(0,3)= 4;  product_table(1,3)= 3;  product_table(2,3)= 2;  product_table(3,3)= 1;
  }

  else if ( pg == (string)"c2h" )
  {
       // C2h Ag: 1, Au: 2, Bu: 3, Bg: 4 in Molrpro. Indices here: Ag: 0, Au: 1, Bu: 2, Bg: 3
    pointGroup = pg;
    product_table.resize(4,4);
    product_table(0,0)= 1;  product_table(1,0)= 2;  product_table(2,0)= 3;  product_table(3,0)= 4;
    product_table(0,1)= 2;  product_table(1,1)= 1;  product_table(2,1)= 4;  product_table(3,1)= 3;
    product_table(0,2)= 3;  product_table(1,2)= 4;  product_table(2,2)= 1;  product_table(3,2)= 2;
    product_table(0,3)= 4;  product_table(1,3)= 3;  product_table(2,3)= 2;  product_table(3,3)= 1;
  }

  else if ( pg == (string)"d2" )
  {
    // D2 A:1, B3:2, B2: 3, B1: 4 in Molrpro. Indices here: A:0, B3:1, B2: 2, B1: 3
    pointGroup = pg;
    product_table.resize(4,4);
    product_table(0,0)= 1;  product_table(1,0)= 2;  product_table(2,0)= 3;  product_table(3,0)= 4;
    product_table(0,1)= 2;  product_table(1,1)= 1;  product_table(2,1)= 4;  product_table(3,1)= 3;
    product_table(0,2)= 3;  product_table(1,2)= 4;  product_table(2,2)= 1;  product_table(3,2)= 2;
    product_table(0,3)= 4;  product_table(1,3)= 3;  product_table(2,3)= 2;  product_table(3,3)= 1;
  }

  else if ( pg == (string)"cs" )
  {
    // Cs A':1, A'':2 in MolPro. Indices here A':0, A'':1
    pointGroup = pg;
    product_table.resize(2,2);
    product_table(0,0)= 1;  product_table(1,0)= 2;
    product_table(0,1)= 2;  product_table(1,1)= 1;
  }

  else if ( pg == (string)"c2" )
  {
    // C2 A:1, B:2 in MolPro. Indices here: A:0, B:1
    pointGroup = pg;
    product_table.resize(2,2);
    product_table(0,0)= 1;  product_table(1,0)= 2;
    product_table(0,1)= 2;  product_table(1,1)= 1;
  }

  else if ( pg == (string)"ci" )
  {
    // Ci Ag:1, Au:2 in MolPro. Indices here: Ag:0, Au:1
    pointGroup = pg;
    product_table.resize(2,2);
    product_table(0,0)= 1;  product_table(1,0)= 2;
    product_table(0,1)= 2;  product_table(1,1)= 1;
  }

else
  {
    cout << "Couldn't find the irrep! Exiting..." << endl;
    exit(EXIT_FAILURE);
  }
    
}


int symmetry::getProduct(int irrep1, int irrep2)
{
  // Note here that the irrep and column differ by 1 because the indexing starts
  // at 0 for the product_table object.
  return product_table(irrep1 - 1 ,irrep2 - 1);  
}


int symmetry::getSymmetry( char* repArray, vector<int>& irrep)
{
  // Returns the irrep of the determinant with the symmetry of pointGroup.
  int norbs = sizeof(irrep) / sizeof(irrep[0]) - 1;
  int old_irrep = 1;
   
  for ( int i = 0; i < norbs; i++ ) 
  {   
    if ( (int) repArray[i] - (int) '0' == 1 )// TODO Watch out for this.
    {
      old_irrep = getProduct( old_irrep, irrep[i] ); 
    }
  }

  return old_irrep;
};

/* MAIN */
int main()
{
  char* repArray="10111";
  //repArray[0] = 1; repArray[1]=0; repArray[2]=1; repArray[3]=1; repArray[4]=1;
  
  size_t size = 5;
  vector<int> irrep(size);
  irrep[0] = 1; irrep[1] = 4; irrep[2]=2; irrep[3]=2; irrep[4]=3;

  symmetry mol_sym((string)"d2h");
	symmetry mol_sym1((string)"c2v");
  cout << "D2h product: " << mol_sym.getSymmetry(repArray, irrep) << endl;
  cout << "C2v product: " << mol_sym1.getSymmetry(repArray,irrep) << endl;
	return 0;

}
