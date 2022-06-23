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
#ifndef SOCHELPER_H
#define SOCHELPER_H

#include <vector>
#include "global.h"

using namespace std;
using namespace Eigen;
class Determinant;

namespace SOChelper {
  void calculateSpinRDM(vector<MatrixXx>& spinRDM, MatrixXx& ci1, MatrixXx& ci2,
			Determinant* Dets1, int Detssize, int norbs, int nelec);

  void getSplus(const MatrixXx& c2, MatrixXx& c2splus,
		vector<Determinant>& Dets, vector<Determinant>::iterator& beginS0,
		vector<Determinant>::iterator& beginSp,
		vector<Determinant>::iterator& beginSm, int norbs) ;

    void getSminus(const MatrixXx& c2, MatrixXx& c2sminus,
		   vector<Determinant>& Dets, vector<Determinant>::iterator& beginS0,
		   vector<Determinant>::iterator& beginSp,
		   vector<Determinant>::iterator& beginSm, int norbs) ;

    void calculateMatrixElements(int spin1, int spin2, int Sz, int rowIndex1, int rowIndex2,
				 const MatrixXx& c1, const MatrixXx& c2,
				 vector<vector<int> >& connections,
				 vector<vector<CItype> >& Helements, MatrixXx& Hsubspace,
				 vector<Determinant>& Dets, int norbs,
				 vector<Determinant>::iterator& beginS0,
				 vector<Determinant>::iterator& beginSp,
				 vector<Determinant>::iterator& beginSm) ;

    void calculateMatrixElementsForgTensor(int spin1, int spin2, int Sz, int rowIndex1, int rowIndex2,
				 const MatrixXx& c1, const MatrixXx& c2,
				 vector<vector<int> >& connections,
				 vector<vector<CItype> >& Helements, vector<MatrixXx>& Hsubspace,
				 vector<Determinant>& Dets, int norbs,
				 vector<Determinant>::iterator& beginS0,
				 vector<Determinant>::iterator& beginSp,
				 vector<Determinant>::iterator& beginSm) ;

    void doGTensor(vector<MatrixXx>& ci, vector<Determinant>& Dets,
		   vector<double>& E0, int norbs, int nelec);
    void doGTensor(vector<MatrixXx>& ci, Determinant* Dets,
		   vector<double>& E0, int Detssize, int norbs, 
		   int nelec, vector<MatrixXx>& spinRDM);
};

#endif
