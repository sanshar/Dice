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
#include "Wfn.h"
#include "integral.h"
#include "CPS.h"
#include "MoDeterminants.h"
using namespace Eigen;

void CPSSlater::printVariables() {
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) {
    for (int j=0; j<cpsArray[i].Variables.size(); j++) {
      cout << "  "<<cpsArray[i].Variables[j];
      numVars++;
    }
  }
  cout <<endl;
}

void CPSSlater::incrementVariables(Eigen::VectorXd& dv){
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) {
    for (int j=0; j<cpsArray[i].Variables.size(); j++) {
      cpsArray[i].Variables[j] += dv[numVars];
      numVars++;
    }
  }
}

void CPSSlater::updateVariables(Eigen::VectorXd& v){
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) {
    for (int j=0; j<cpsArray[i].Variables.size(); j++) {
      cpsArray[i].Variables[j] = v[numVars];
      numVars++;
    }
  }
}

void CPSSlater::getVariables(Eigen::VectorXd& v){
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) {
    for (int j=0; j<cpsArray[i].Variables.size(); j++) {
      v[numVars] = cpsArray[i].Variables[j];
      numVars++;
    }
  }
}

long CPSSlater::getNumVariables() {
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) 
    numVars += cpsArray[i].Variables.size();
  return numVars;
}

void CPSSlater::OverlapWithGradient(Determinant& d, 
				    double& factor,
				    VectorXd& grad) {
  double ovlp = Overlap(d)*factor;
  long startIndex = 0;

  for (int i=0; i<cpsArray.size(); i++) {
    cpsArray[i].OverlapWithGradient(d, grad, 
				    ovlp, startIndex);
    startIndex += cpsArray[i].Variables.size();
  }
}

double CPSSlater::Overlap(Determinant& d) {
  double ovlp=1.0;
  for (int i=0; i<cpsArray.size(); i++) {
    ovlp *= cpsArray[i].Overlap(d);
  }
  return ovlp*det.Overlap(d);

}


void CPSSlater::HamAndOvlp(Determinant& d,
			   double& ovlp, double& ham,
			   oneInt& I1, twoInt& I2, double& coreE) {

  //cout << d <<endl;
  //noexcitation
  {
    double E0 = d.Energy(I1, I2, coreE);
    ovlp = det.Overlap(d);
    for (int i=0; i<cpsArray.size(); i++)
      ovlp *= cpsArray[i].Overlap(d);
    ham  = ovlp*E0;
    //cout << E0<<"  "<<ovlp<<"  "<<ham<<" - ";
  }

  //Single excitation 
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getocc(i)) {
	for (int a=0; a<Determinant::norbs; a++)
	  if (!d.getocc(a)) {
	    double tia = d.Hij_1Excite(a, i, I1, I2);
	    double localham = 0.0;
	    if (abs(tia) > 1.e-8) {
	      Determinant dcopy = d;
	      dcopy.setocc(i, false); dcopy.setocc(a, true);
	      localham += tia*det.Overlap(dcopy);
	      for (int i=0; i<cpsArray.size(); i++)
		localham *= cpsArray[i].Overlap(dcopy);
	      ham += localham;
	      //cout <<dcopy<<"  "<< tia<<"  "<<localham<<"  "<<ham<<" - ";
	    }

	  }
      }
  }

  //Single excitation alpha
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getocc(i)) {
	for (int j=i+1; j<Determinant::norbs; j++) 
	  if (d.getocc(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getocc(a)) {
		for (int b=a+1; b<Determinant::norbs; b++)
		  if (!d.getocc(b)) {

		    double tiajb = d.Hij_2Excite(i, j, a, b, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-8) {
		      Determinant dcopy = d;
		      dcopy.setocc(i, false); dcopy.setocc(a, true);
		      dcopy.setocc(j, false); dcopy.setocc(b, true);
		      localham += tiajb*det.Overlap(dcopy);
		      for (int i=0; i<cpsArray.size(); i++)
			localham *= cpsArray[i].Overlap(dcopy);
		      ham += localham;
		      //cout <<dcopy<<"  "<< tiajb<<"  "<<localham<<"  "<<ham<<" - ";
		    }
		  }
	      }
	  }
      }
  }
}




void CPSUniform::printVariables() {
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) {
    for (int j=0; j<cpsArray[i].Variables.size(); j++) {
      cout << "  "<<cpsArray[i].Variables[j];
      numVars++;
    }
  }
  cout <<endl;
}

void CPSUniform::incrementVariables(Eigen::VectorXd& dv){
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) {
    for (int j=0; j<cpsArray[i].Variables.size(); j++) {
      cpsArray[i].Variables[j] += dv[numVars];
      numVars++;
    }
  }
}

void CPSUniform::updateVariables(Eigen::VectorXd& v){
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) {
    for (int j=0; j<cpsArray[i].Variables.size(); j++) {
      cpsArray[i].Variables[j] = v[numVars];
      numVars++;
    }
  }
}

void CPSUniform::getVariables(Eigen::VectorXd& v){
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) {
    for (int j=0; j<cpsArray[i].Variables.size(); j++) {
      v[numVars] = cpsArray[i].Variables[j];
      numVars++;
    }
  }
}

long CPSUniform::getNumVariables() {
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) 
    numVars += cpsArray[i].Variables.size();
  return numVars;
}

void CPSUniform::OverlapWithGradient(Determinant& d, 
				    double& factor,
				    VectorXd& grad) {
  double ovlp = Overlap(d)*factor;
  long startIndex = 0;

  for (int i=0; i<cpsArray.size(); i++) {
    cpsArray[i].OverlapWithGradient(d, grad, 
				    ovlp, startIndex);
    startIndex += cpsArray[i].Variables.size();
  }
}

double CPSUniform::Overlap(Determinant& d) {
  double ovlp=1.0;
  for (int i=0; i<cpsArray.size(); i++) {
    ovlp *= cpsArray[i].Overlap(d);
  }
  return ovlp;
}


void CPSUniform::HamAndOvlp(Determinant& d,
			   double& ovlp, double& ham,
			   oneInt& I1, twoInt& I2, double& coreE) {

  //cout << d <<endl;
  //noexcitation
  {
    double E0 = d.Energy(I1, I2, coreE);
    ovlp = 1.0;
    for (int i=0; i<cpsArray.size(); i++)
      ovlp *= cpsArray[i].Overlap(d);
    ham  = ovlp*E0;
    //cout << E0<<"  "<<ovlp<<"  "<<ham<<" - ";
  }

  //Single excitation 
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getocc(i)) {
	for (int a=0; a<Determinant::norbs; a++)
	  if (!d.getocc(a)) {
	    double tia = d.Hij_1Excite(a, i, I1, I2);
	    double localham = 0.0;
	    if (abs(tia) > 1.e-8) {
	      Determinant dcopy = d;
	      dcopy.setocc(i, false); dcopy.setocc(a, true);
	      localham += tia;
	      for (int i=0; i<cpsArray.size(); i++)
		localham *= cpsArray[i].Overlap(dcopy);
	      ham += localham;
	      //cout <<dcopy<<"  "<< tia<<"  "<<localham<<"  "<<ham<<" - ";
	    }

	  }
      }
  }

  //Single excitation alpha
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getocc(i)) {
	for (int j=i+1; j<Determinant::norbs; j++) 
	  if (d.getocc(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getocc(a)) {
		for (int b=a+1; b<Determinant::norbs; b++)
		  if (!d.getocc(b)) {

		    double tiajb = d.Hij_2Excite(i, j, a, b, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-8) {
		      Determinant dcopy = d;
		      dcopy.setocc(i, false); dcopy.setocc(a, true);
		      dcopy.setocc(j, false); dcopy.setocc(b, true);
		      localham += tiajb;
		      for (int i=0; i<cpsArray.size(); i++)
			localham *= cpsArray[i].Overlap(dcopy);
		      ham += localham;
		      //cout <<dcopy<<"  "<< tiajb<<"  "<<localham<<"  "<<ham<<" - ";
		    }
		  }
	      }
	  }
      }
  }
}

