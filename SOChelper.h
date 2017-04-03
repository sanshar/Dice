#ifndef SOCHELPER_H
#define SOCHELPER_H

#include <vector>
#include "global.h"

using namespace std;
using namespace Eigen;
class Determinant;

namespace SOChelper {
  void calculateSpinRDM(vector<MatrixXx>& spinRDM, MatrixXx& ci1, MatrixXx& ci2, 
			vector<Determinant>& Dets1, int norbs, int nelec);

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

    void doGTensor(vector<MatrixXx>& ci, vector<Determinant>& Dets, 
		   vector<double>& E0, int norbs, int nelec);
    void doGTensor(vector<MatrixXx>& ci, vector<Determinant>& Dets, 
		   vector<double>& E0, int norbs, int nelec, vector<MatrixXx>& spinRDM);
};

#endif
