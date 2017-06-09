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
#include "global.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <boost/bind.hpp>
#include <Eigen/Dense>

#include "integral.h"
#include "Determinants.h"

using namespace Eigen;
using namespace std;

// class oneInt;
// class Determinant;

bool compare(const pair<double,int>& a,const pair<double,int>& b) {
	return a.second<b.second;
}

class symmetry {
private:
	// Full table for d2h and all subgroups
	MatrixXd fullD2hTable;

public:
	static MatrixXd product_table;       //product table for a given symmetry
	string pointGroup;       //eg. d2h, c2
	symmetry(string);       //product and sym is initialized
	int getProduct(int,int);
	int getProduct(vector<int>&);
	int getSymmetry(char*, vector<int>&);
	void estimateLowestEnergyDet( int, int, oneInt, vector<int>&,
	  vector<vector<int> >&, Determinant& );
};

symmetry::symmetry( string pg ) {

	// Assign correct product table for point group. Note that even though the
	// indices of the columns are one less than the MolPro notation the irrep
	// returned is again in the MolPro notation.

	// D2h Ag:1, B1g:4, B2g:6, B3g:7, Au:8, B1u:5, B2u:3, B3u: 2 in MolPro.
	// Row & col indices: Ag:0, B1g:3, B2g:5, B3g:6, Au:7, B1u:4, B2u:2, B3u:1
	fullD2hTable.resize(8,8);
	fullD2hTable(0,0) = 1; fullD2hTable(1,0) = 2; fullD2hTable(2,0) = 3;
	fullD2hTable(0,1) = 2; fullD2hTable(1,1) = 1; fullD2hTable(2,1) = 4;
	fullD2hTable(0,2) = 3; fullD2hTable(1,2) = 4; fullD2hTable(2,2) = 1;
	fullD2hTable(0,3) = 4; fullD2hTable(1,3) = 3; fullD2hTable(2,3) = 2;
	fullD2hTable(0,4) = 5; fullD2hTable(1,4) = 6; fullD2hTable(2,4) = 7;
	fullD2hTable(0,5) = 6; fullD2hTable(1,5) = 5; fullD2hTable(2,5) = 8;
	fullD2hTable(0,6) = 7; fullD2hTable(1,6) = 8; fullD2hTable(2,6) = 5;
	fullD2hTable(0,7) = 8; fullD2hTable(1,7) = 7; fullD2hTable(2,7) = 6;

	fullD2hTable(3,0) = 4; fullD2hTable(4,0) = 5; fullD2hTable(5,0) = 6;
	fullD2hTable(3,1) = 3; fullD2hTable(4,1) = 6; fullD2hTable(5,1) = 5;
	fullD2hTable(3,2) = 2; fullD2hTable(4,2) = 7; fullD2hTable(5,2) = 8;
	fullD2hTable(3,3) = 1; fullD2hTable(4,3) = 8; fullD2hTable(5,3) = 7;
	fullD2hTable(3,4) = 8; fullD2hTable(4,4) = 1; fullD2hTable(5,4) = 2;
	fullD2hTable(3,5) = 7; fullD2hTable(4,5) = 2; fullD2hTable(5,5) = 1;
	fullD2hTable(3,6) = 6; fullD2hTable(4,6) = 3; fullD2hTable(5,6) = 4;
	fullD2hTable(3,7) = 5; fullD2hTable(4,7) = 4; fullD2hTable(5,7) = 3;

	fullD2hTable(6,0) = 7; fullD2hTable(7,0) = 8;
	fullD2hTable(6,1) = 8; fullD2hTable(7,1) = 7;
	fullD2hTable(6,2) = 5; fullD2hTable(7,2) = 6;
	fullD2hTable(6,3) = 6; fullD2hTable(7,3) = 5;
	fullD2hTable(6,4) = 3; fullD2hTable(7,4) = 4;
	fullD2hTable(6,5) = 4; fullD2hTable(7,5) = 3;
	fullD2hTable(6,6) = 1; fullD2hTable(7,6) = 2;
	fullD2hTable(6,7) = 2; fullD2hTable(7,7) = 1;

	if ( pg == (string)"d2h" ) {
		pointGroup = pg;
		product_table = fullD2hTable;
	}

	else if ( pg == (string)"c2v" ) {
		//C2v A1: 1, B1:2, B2:3, A2:4 in MolPro. Indices here: A1: 0, B1:1, B2:2, A2:3
		pointGroup = pg;
		product_table.resize(4,4);
		product_table = (fullD2hTable.topRows(4)).leftCols(4);
	}

	else if ( pg == (string)"c2h" ) {
		// C2h Ag: 1, Au: 2, Bu: 3, Bg: 4 in Molrpro. Indices here: Ag: 0, Au: 1, Bu: 2, Bg: 3
		pointGroup = pg;
		product_table.resize(4,4);
		product_table = (fullD2hTable.topRows(4)).leftCols(4);
	}

	else if ( pg == (string)"d2" ) {
		// D2 A:1, B3:2, B2: 3, B1: 4 in Molrpro. Indices here: A:0, B3:1, B2: 2, B1: 3
		pointGroup = pg;
		product_table.resize(4,4);
		product_table = (fullD2hTable.topRows(4)).leftCols(4);
	}

	else if ( pg == (string)"cs" ) {
		// Cs A':1, A'':2 in MolPro. Indices here A':0, A'':1
		pointGroup = pg;
		product_table.resize(2,2);
		product_table = (fullD2hTable.topRows(2)).leftCols(2);
	}

	else if ( pg == (string)"c2" ) {
		// C2 A:1, B:2 in MolPro. Indices here: A:0, B:1
		pointGroup = pg;
		product_table.resize(2,2);
		product_table = (fullD2hTable.topRows(2)).leftCols(2);
	}

	else if ( pg == (string)"ci" ) {
		// Ci Ag:1, Au:2 in MolPro. Indices here: Ag:0, Au:1
		pointGroup = pg;
		product_table.resize(2,2);
		product_table = (fullD2hTable.topRows(2)).leftCols(2);
	}

	else {
		cout << "Couldn't find the irrep! Exiting..." << endl;
		exit(EXIT_FAILURE);
	}

}


int symmetry::getProduct(int irrep1, int irrep2) {
	// Note here that the irrep and column differ by 1 because the indexing starts
	// at 0 for the product_table object.
	return product_table(irrep1 - 1,irrep2 - 1);
}

int symmetry::getProduct(vector<int>& irreps) { //TODO test this
	// For more than two irreps.
	if ( irreps.size() > 2 ) {
		int irrep = irreps.back();
		irreps.pop_back();
		return getProduct( getProduct( irreps ), irrep );
	}

	else { return getProduct( irreps[0], irreps[1] ); }

}


int symmetry::getSymmetry( char* repArray, vector<int>& irrep) {
	// Returns the irrep of the determinant with the symmetry of pointGroup.
	int norbs = sizeof(irrep) / sizeof(irrep[0]) - 1;
	int old_irrep = 1;

	for ( int i = 0; i < norbs; i++ ) {
		if ( (int) repArray[i] - (int) '0' == 1 ) { // TODO Watch out for this.
			old_irrep = getProduct( old_irrep, irrep[i] );
		}
	}

	return old_irrep;
}

void symmetry::estimateLowestEnergyDet( int spin, int targetIrrep, oneInt I1,
  vector<int>& irrep, vector<vector<int> >& occupied, Determinant& Det ) {

	// This method is for a single determinant and should be placed inside a loop
	// over all input desired determinants.

	vector<pair<double,int> > sort1Body (0);  // Double: E, int: original idx
	for ( int i=0; i < I1.norbs; i++ ) {
		sort1Body.push_back(make_pair( I1(i,i), i ));
	}
	sort(sort1Body.begin(),sort1Body.end(),compare);

	// Populate doubly occupied orbitals
	int nDOrbs = occupied[0].size() - spin;
	for ( int i=0; i < nDOrbs; i++ ) { Det.setocc(sort1Body[i].second, true); }

	// Spin dependent population of remaining singly occupied orbitals.
	if ( spin == 0 ) {
		return;
	}

	else if ( spin == 1 ) {
		// Find lowest energy orbital with targetIrrep
		for ( int i = nDOrbs; i < I1.norbs; i++ ) {
			if ( irrep[sort1Body[i].second] == targetIrrep &&
			  (Det.getocc(i) == false) ) {
				Det.setocc( i, true );
				return;
			}
		}
	}

	else if ( spin == 2 ) {
		// Find lowest energy orbitals with the appropriate symmetry
		for ( int i=nDOrbs; i < I1.norbs - 1; i++ ) {
			for ( int j=i+1; j < I1.norbs; j++ ) {
				int irrep1 = irrep[sort1Body[i].second];
				int irrep2 = irrep[sort1Body[j].second];
				bool unocc = ( (Det.getocc(i) == false) &&
				  (Det.getocc(j) == false) );

				if ( getProduct( irrep1, irrep2 ) == targetIrrep && unocc ) {
					Det.setocc( i, true );
					Det.setocc( j, true );
					return;
				}
			}
		}
	}

	else if ( spin == 3 ) {
		// Find lowest energy orbitals with the appropriate symmetry
		for ( int i=nDOrbs; i < I1.norbs - 2; i++ ) {
			for ( int j=i+1; j < I1.norbs - 1; j++ ) {
				for ( int k=j+1; k < I1.norbs; k++ ) {
					vector<int> irreps (3);
					irreps[0] = irrep[sort1Body[i].second];
					irreps[1] = irrep[sort1Body[j].second];
					irreps[2] = irrep[sort1Body[k].second];
					bool unocc = ( (Det.getocc(i) == false) &&
					  (Det.getocc(j) == false) && (Det.getocc(k) == false) );

					if ( getProduct( irreps ) == targetIrrep && unocc )
					{
						Det.setocc( i, true );
						Det.setocc( j, true );
						Det.setocc( k, true );
						return;
					}
				}
			}
		}
	}

	else if ( spin == 4 ) {
		// Find lowest energy orbitals with the appropriate symmetry
		for ( int i=nDOrbs; i < I1.norbs - 3; i++ ) {
			for ( int j=i+1; j < I1.norbs - 2; j++ ) {
				for ( int k=j+1; k < I1.norbs - 1; k++ ) {
					for ( int l=k+1; l < I1.norbs; l++ ) {
						vector<int> irreps (4);
						irreps[0] = irrep[sort1Body[i].second];
						irreps[1] = irrep[sort1Body[j].second];
						irreps[2] = irrep[sort1Body[k].second];
						irreps[3] = irrep[sort1Body[l].second];
						bool unocc = ( (Det.getocc(i) == false) &&
						  (Det.getocc(j) == false) && (Det.getocc(k) == false)  &&
						  (Det.getocc(k) == false) );

						if ( getProduct( irreps ) == targetIrrep && unocc )
						{
							Det.setocc( i, true );
							Det.setocc( j, true );
							Det.setocc( k, true );
							Det.setocc( l, true );
							return;
						}
					}
				}
			}
		}
	}

	else {
		printf ("Spin currently not supported by Dice. Please contact authors");
		exit (EXIT_FAILURE);
	}
};
