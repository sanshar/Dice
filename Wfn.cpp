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
#include "Walker.h"
#include "Wfn.h"
#include "global.h"
#include "input.h"
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;

void CPSSlater::getDetMatrix(Determinant& d, Eigen::MatrixXd& alpha, Eigen::MatrixXd& beta) {
  det.getDetMatrix(d, alpha, beta);
}

void CPSSlater::printVariables() {
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) {
    for (int j=0; j<cpsArray[i].Variables.size(); j++) {
      cout << "  "<<cpsArray[i].Variables[j];
      numVars++;
    }
  }

  for (int i=0; i<det.AlphaOrbitals.rows(); i++)
    for (int j=0; j<det.AlphaOrbitals.cols(); j++) {
      cout <<"  "<<det.AlphaOrbitals(i,j);
      numVars++;
    }

  for (int i=0; i<det.BetaOrbitals.rows(); i++)
    for (int j=0; j<det.BetaOrbitals.cols(); j++) {
      cout <<"  "<<det.BetaOrbitals(i,j);
      numVars++;
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


  for (int i=0; i<det.AlphaOrbitals.rows(); i++)
    for (int j=0; j<det.AlphaOrbitals.cols(); j++) {
      det.AlphaOrbitals(i,j) += dv[numVars];
      numVars++;
    }

  for (int i=0; i<det.BetaOrbitals.rows(); i++)
    for (int j=0; j<det.BetaOrbitals.cols(); j++) {
      det.BetaOrbitals(i,j) += dv[numVars];
      numVars++;
    }


}

void orthogonalise(MatrixXd& m) {

  for (int i=0; i<m.cols(); i++) {
    for (int j=0; j<i; j++) {
      double ovlp = m.col(i).transpose()*m.col(j) ;
      double norm = m.col(j).transpose()*m.col(j) ;
      m.col(i) = m.col(i)- ovlp/norm*m.col(j);
    }
    m.col(i) =  m.col(i)/pow(m.col(i).transpose()*m.col(i), 0.5);
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

  /*
  for (int i=0; i<det.AlphaOrbitals.rows(); i++)
    for (int j=0; j<det.AlphaOrbitals.cols(); j++) {
      det.AlphaOrbitals(i,j) = v[numVars];
      numVars++;
    }

  orthogonalise(det.AlphaOrbitals);

  for (int i=0; i<det.BetaOrbitals.rows(); i++)
    for (int j=0; j<det.BetaOrbitals.cols(); j++) {
      det.BetaOrbitals(i,j) = v[numVars];
      numVars++;
    }
  orthogonalise(det.BetaOrbitals);
  */
}

void CPSSlater::getVariables(Eigen::VectorXd& v){
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) {
    for (int j=0; j<cpsArray[i].Variables.size(); j++) {
      v[numVars] = cpsArray[i].Variables[j];
      numVars++;
    }
  }


  for (int i=0; i<det.AlphaOrbitals.rows(); i++)
    for (int j=0; j<det.AlphaOrbitals.cols(); j++) {
      v[numVars] = det.AlphaOrbitals(i,j);
      numVars++;
    }

  for (int i=0; i<det.BetaOrbitals.rows(); i++)
    for (int j=0; j<det.BetaOrbitals.cols(); j++) {
      v[numVars] = det.BetaOrbitals(i,j);
      numVars++;
    }
}

long CPSSlater::getNumVariables() {
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) 
    numVars += cpsArray[i].Variables.size();
  numVars+=det.norbs*det.nalpha+det.norbs*det.nbeta;
  return numVars;
}

//factor = <psi|w> * prefactor;
void CPSSlater::OverlapWithGradient(Walker& w, 
				    double& factor,
				    VectorXd& grad) {
  //double ovlp = w.alphaDet*w.betaDet*factor;
  Determinant& d = w.d;
  long startIndex = 0;

  for (int i=0; i<cpsArray.size(); i++) {
    cpsArray[i].OverlapWithGradient(d, grad, 
				    factor, startIndex);
    startIndex += cpsArray[i].Variables.size();
  }
  /*
  int aindex = 0;
  for (int i=0; i<Determinant::norbs; i++) {
    if (d.getoccA(i)) {
      for (int j=0; j<det.AlphaOrbitals.cols(); j++) {
	grad[startIndex] += ovlp*alphainv(j, aindex);
      }
      aindex++;
    }
    startIndex++;
  }

  int bindex = 0;
  for (int i=0; i<Determinant::norbs; i++) {
    if (d.getoccB(i)) {
      for (int j=0; j<det.BetaOrbitals.cols(); j++) {
	grad[startIndex] += ovlp*betainv(j, bindex);
      }
      bindex++;
    }
    startIndex++;
  }
  */
}

void CPSSlater::OverlapWithGradient(Determinant& d, 
				    double& factor,
				    VectorXd& grad) {
  //double ovlp = Overlap(d)*factor;
  double ovlp = factor;
  long startIndex = 0;

  //****
  //MatrixXd alpha, beta;
  //det.getDetMatrix(d, alpha, beta);
  //MatrixXd alphainv = alpha.inverse(), betainv = beta.inverse();
  //****

  for (int i=0; i<cpsArray.size(); i++) {
    cpsArray[i].OverlapWithGradient(d, grad, 
				    ovlp, startIndex);
    startIndex += cpsArray[i].Variables.size();
  }
  /*
  int aindex = 0;
  for (int i=0; i<Determinant::norbs; i++) {
    if (d.getoccA(i)) {
      for (int j=0; j<det.AlphaOrbitals.cols(); j++) {
	grad[startIndex] += ovlp*alphainv(j, aindex);
      }
      aindex++;
    }
    startIndex++;
  }

  int bindex = 0;
  for (int i=0; i<Determinant::norbs; i++) {
    if (d.getoccB(i)) {
      for (int j=0; j<det.BetaOrbitals.cols(); j++) {
	grad[startIndex] += ovlp*betainv(j, bindex);
      }
      bindex++;
    }
    startIndex++;
  }
  */
}

void CPSSlater::writeWave() {
  if (commrank == 0) {
    char file [5000];
    //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
    sprintf (file, "wave.bkp");
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << *this;
    ofs.close();
  }
}

void CPSSlater::readWave() {
  if (commrank == 0) {
    char file [5000];
    //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
    sprintf (file, "wave.bkp");
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    load >> *this;
    ifs.close();
  }
#ifndef SERIAL
  boost::mpi::communicator world;
  boost::mpi::broadcast(world, *this, 0);
#endif
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

  MatrixXd alpha, beta;
  det.getDetMatrix(d, alpha, beta);
  MatrixXd alphainv = alpha.inverse(), betainv = beta.inverse();
  double alphaDet = alpha.determinant(), betaDet = beta.determinant();
  double detOverlap = alphaDet*betaDet;

  //noexcitation
  {
    double E0 = d.Energy(I1, I2, coreE);
    ovlp = detOverlap;
    for (int i=0; i<cpsArray.size(); i++)
      ovlp *= cpsArray[i].Overlap(d);
    ham  = ovlp*E0;
    //detOverlap *= cpsOvlp;
  }

  
  //Single alpha excitation 
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccA(i)) {
	for (int a=0; a<Determinant::norbs; a++)
	  if (!d.getoccA(a)) {
	    double tia = d.Hij_1ExciteA(a, i, I1, I2);
	    double localham = 0.0;
	    if (abs(tia) > 1.e-10) {
	      Determinant dcopy = d;
	      dcopy.setoccA(i, false); dcopy.setoccA(a, true);

	      localham += tia*det.OverlapA(d, i, a,  alphainv, betainv)*ovlp;

	      for (int n=0; n<cpsArray.size(); n++) 
		if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), i) ||
		    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), a) )
		  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);
	      ham += localham;
	    }

	  }
      }
  }


  //Single beta excitation 
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccB(i)) {
	for (int a=0; a<Determinant::norbs; a++)
	  if (!d.getoccB(a)) {
	    double tia = d.Hij_1ExciteB(a, i, I1, I2);
	    double localham = 0.0;
	    if (abs(tia) > 1.e-10) {
	      Determinant dcopy = d;
	      dcopy.setoccB(i, false); dcopy.setoccB(a, true);

	      localham += tia*det.OverlapB(d, i, a,  alphainv, betainv)*ovlp;

	      for (int n=0; n<cpsArray.size(); n++) 
		if (std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), i) ||
		    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), a) )
		  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

	      ham += localham;

	    }

	  }
      }
  }


  //Double excitation alpha alpha
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccA(i)) {
	for (int j=i+1; j<Determinant::norbs; j++) 
	  if (d.getoccA(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getoccA(a)) {
		for (int b=a+1; b<Determinant::norbs; b++)
		  if (!d.getoccA(b)) {

		    double tiajb = d.Hij_2ExciteAA(a, i, b, j, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-10) {
		      Determinant dcopy = d;
		      dcopy.setoccA(i, false); dcopy.setoccA(a, true);
		      dcopy.setoccA(j, false); dcopy.setoccA(b, true);

		      localham += tiajb*det.OverlapAA(d, i, j, a, b,  alphainv, betainv)
			*ovlp;

		      for (int n=0; n<cpsArray.size(); n++) 
			if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), i) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), j) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), a) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), b) )
			  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

		      ham += localham;
		    }
		  }
	      }
	  }
      }
  }


  //Double excitation beta beta
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccB(i)) {
	for (int j=i+1; j<Determinant::norbs; j++) 
	  if (d.getoccB(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getoccB(a)) {
		for (int b=a+1; b<Determinant::norbs; b++)
		  if (!d.getoccB(b)) {

		    double tiajb = d.Hij_2ExciteBB(a, i, b, j, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-10) {
		      Determinant dcopy = d;
		      dcopy.setoccB(i, false); dcopy.setoccB(a, true);
		      dcopy.setoccB(j, false); dcopy.setoccB(b, true);

		      localham += tiajb*det.OverlapBB(d, i, j, a, b,  alphainv, betainv)
			*ovlp;

		      for (int n=0; n<cpsArray.size(); n++) 
			if (std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), i) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), j) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), a) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), b) )
			  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

		      ham += localham;
		      //cout <<dcopy<<"  "<< tiajb<<"  "<<localham<<"  "<<ham<<" - ";
		    }
		  }
	      }
	  }
      }
  }


  //Double excitation alpha beta
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccA(i)) {
	for (int j=0; j<Determinant::norbs; j++) 
	  if (d.getoccB(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getoccA(a)) {
		for (int b=0; b<Determinant::norbs; b++)
		  if (!d.getoccB(b)) {

		    double tiajb = d.Hij_2ExciteAB(a, i, b, j, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-10) {
		      Determinant dcopy = d;
		      dcopy.setoccA(i, false); dcopy.setoccA(a, true);
		      dcopy.setoccB(j, false); dcopy.setoccB(b, true);

		      localham += tiajb*det.OverlapAB(d, i, j, a, b,  alphainv, betainv)
			*ovlp;

		      for (int n=0; n<cpsArray.size(); n++) 
			if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), i) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), j) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), a) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), b) )
			  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

		      ham += localham;
		      //cout <<dcopy<<"  "<< tiajb<<"  "<<localham<<"  "<<ham<<" - ";
		    }
		  }
	      }
	  }
      }
  }

}

void CPSSlater::HamAndOvlp(Walker& walk,
			   double& ovlp, double& ham,
			   oneInt& I1, twoInt& I2, double& coreE) {

  MatrixXd alphainv = walk.alphainv, betainv = walk.betainv;
  double alphaDet = walk.alphaDet, betaDet = walk.betaDet;
  double detOverlap = alphaDet*betaDet;
  Determinant& d = walk.d;

  //noexcitation
  {
    double E0 = d.Energy(I1, I2, coreE);
    ovlp = detOverlap;
    for (int i=0; i<cpsArray.size(); i++)
      ovlp *= cpsArray[i].Overlap(d);
    ham  = ovlp*E0;
    //detOverlap *= cpsOvlp;
  }

  
  //Single alpha excitation 
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccA(i)) {
	for (int a=0; a<Determinant::norbs; a++)
	  if (!d.getoccA(a)) {
	    double tia = d.Hij_1ExciteA(a, i, I1, I2);
	    double localham = 0.0;
	    if (abs(tia) > 1.e-10) {
	      Determinant dcopy = d;
	      dcopy.setoccA(i, false); dcopy.setoccA(a, true);

	      localham += tia*det.OverlapA(d, i, a,  alphainv, betainv)*ovlp;

	      for (int n=0; n<cpsArray.size(); n++) 
		if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), i) ||
		    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), a) )
		  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);
	      ham += localham;
	    }

	  }
      }
  }


  //Single beta excitation 
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccB(i)) {
	for (int a=0; a<Determinant::norbs; a++)
	  if (!d.getoccB(a)) {
	    double tia = d.Hij_1ExciteB(a, i, I1, I2);
	    double localham = 0.0;
	    if (abs(tia) > 1.e-10) {
	      Determinant dcopy = d;
	      dcopy.setoccB(i, false); dcopy.setoccB(a, true);

	      localham += tia*det.OverlapB(d, i, a,  alphainv, betainv)*ovlp;

	      for (int n=0; n<cpsArray.size(); n++) 
		if (std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), i) ||
		    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), a) )
		  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

	      ham += localham;

	    }

	  }
      }
  }


  //Double excitation alpha alpha
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccA(i)) {
	for (int j=i+1; j<Determinant::norbs; j++) 
	  if (d.getoccA(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getoccA(a)) {
		for (int b=a+1; b<Determinant::norbs; b++)
		  if (!d.getoccA(b)) {

		    double tiajb = d.Hij_2ExciteAA(a, i, b, j, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-10) {
		      Determinant dcopy = d;
		      dcopy.setoccA(i, false); dcopy.setoccA(a, true);
		      dcopy.setoccA(j, false); dcopy.setoccA(b, true);

		      localham += tiajb*det.OverlapAA(d, i, j, a, b,  alphainv, betainv)
			*ovlp;

		      for (int n=0; n<cpsArray.size(); n++) 
			if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), i) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), j) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), a) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), b) )
			  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

		      ham += localham;
		    }
		  }
	      }
	  }
      }
  }


  //Double excitation beta beta
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccB(i)) {
	for (int j=i+1; j<Determinant::norbs; j++) 
	  if (d.getoccB(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getoccB(a)) {
		for (int b=a+1; b<Determinant::norbs; b++)
		  if (!d.getoccB(b)) {

		    double tiajb = d.Hij_2ExciteBB(a, i, b, j, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-10) {
		      Determinant dcopy = d;
		      dcopy.setoccB(i, false); dcopy.setoccB(a, true);
		      dcopy.setoccB(j, false); dcopy.setoccB(b, true);

		      localham += tiajb*det.OverlapBB(d, i, j, a, b,  alphainv, betainv)
			*ovlp;

		      for (int n=0; n<cpsArray.size(); n++) 
			if (std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), i) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), j) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), a) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), b) )
			  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

		      ham += localham;
		      //cout <<dcopy<<"  "<< tiajb<<"  "<<localham<<"  "<<ham<<" - ";
		    }
		  }
	      }
	  }
      }
  }


  //Double excitation alpha beta
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccA(i)) {
	for (int j=0; j<Determinant::norbs; j++) 
	  if (d.getoccB(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getoccA(a)) {
		for (int b=0; b<Determinant::norbs; b++)
		  if (!d.getoccB(b)) {

		    double tiajb = d.Hij_2ExciteAB(a, i, b, j, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-10) {
		      Determinant dcopy = d;
		      dcopy.setoccA(i, false); dcopy.setoccA(a, true);
		      dcopy.setoccB(j, false); dcopy.setoccB(b, true);

		      localham += tiajb*det.OverlapAB(d, i, j, a, b,  alphainv, betainv)
			*ovlp;

		      for (int n=0; n<cpsArray.size(); n++) 
			if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), i) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), j) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), a) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), b) )
			  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

		      ham += localham;
		      //cout <<dcopy<<"  "<< tiajb<<"  "<<localham<<"  "<<ham<<" - ";
		    }
		  }
	      }
	  }
      }
  }

}


//<psi_t| (H0-E0)^-1 (H-E0) |D>
void CPSSlater::HamAndOvlpGradient(Walker& walk,
				   double& ovlp, double& ham, VectorXd& grad, double& scale,
				   double& Epsi, oneInt& I1, twoInt& I2, double& coreE) {

  MatrixXd alphainv = walk.alphainv, betainv = walk.betainv;
  double alphaDet = walk.alphaDet, betaDet = walk.betaDet;
  double detOverlap = alphaDet*betaDet;
  Determinant& d = walk.d;

  //noexcitation
  {
    double E0 = d.Energy(I1, I2, coreE);
    ovlp = detOverlap;
    for (int i=0; i<cpsArray.size(); i++)
      ovlp *= cpsArray[i].Overlap(d);
    ham  = ovlp*E0;

    double factor = schd.davidsonPrecondition ? -ovlp*scale : ovlp*(E0-Epsi)*scale;
    OverlapWithGradient(d, factor, grad);
  }

  //Single alpha excitation 
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccA(i)) {
	for (int a=0; a<Determinant::norbs; a++)
	  if (!d.getoccA(a)) {
	    double tia = d.Hij_1ExciteA(a, i, I1, I2);
	    double localham = 0.0;
	    if (abs(tia) > 1.e-10) {
	      Determinant dcopy = d;
	      dcopy.setoccA(i, false); dcopy.setoccA(a, true);

	      localham += tia*det.OverlapA(d, i, a,  alphainv, betainv)*ovlp;

	      for (int n=0; n<cpsArray.size(); n++) 
		if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), i) ||
		    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), a) )
		  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);
	      ham += localham;

	      double Edet = dcopy.Energy(I1, I2, coreE);
	      double ovlpdetcopy = localham/tia;
	      double factor = tia/(Epsi-Edet) * ovlpdetcopy*scale;
	      if( !schd.davidsonPrecondition) factor *= -(Edet-Epsi);
	      OverlapWithGradient(dcopy, factor, grad);

	    }

	  }
      }
  }

  
  //Single beta excitation 
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccB(i)) {
	for (int a=0; a<Determinant::norbs; a++)
	  if (!d.getoccB(a)) {
	    double tia = d.Hij_1ExciteB(a, i, I1, I2);
	    double localham = 0.0;
	    if (abs(tia) > 1.e-10) {
	      Determinant dcopy = d;
	      dcopy.setoccB(i, false); dcopy.setoccB(a, true);

	      localham += tia*det.OverlapB(d, i, a,  alphainv, betainv)*ovlp;

	      for (int n=0; n<cpsArray.size(); n++) 
		if (std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), i) ||
		    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), a) )
		  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

	      ham += localham;

	      double Edet = dcopy.Energy(I1, I2, coreE);
	      double ovlpdetcopy = localham/tia;
	      double factor = tia/(Epsi-Edet) * ovlpdetcopy*scale;
	      if( !schd.davidsonPrecondition) factor *= -(Edet-Epsi);
	      //double factor = tia/(Epsi-Edet);
	      OverlapWithGradient(dcopy, factor, grad);

	    }

	  }
      }
  }

  //Double excitation alpha alpha
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccA(i)) {
	for (int j=i+1; j<Determinant::norbs; j++) 
	  if (d.getoccA(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getoccA(a)) {
		for (int b=a+1; b<Determinant::norbs; b++)
		  if (!d.getoccA(b)) {

		    double tiajb = d.Hij_2ExciteAA(a, i, b, j, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-10) {
		      Determinant dcopy = d;
		      dcopy.setoccA(i, false); dcopy.setoccA(a, true);
		      dcopy.setoccA(j, false); dcopy.setoccA(b, true);

		      localham += tiajb*det.OverlapAA(d, i, j, a, b,  alphainv, betainv)
			*ovlp;

		      for (int n=0; n<cpsArray.size(); n++) 
			if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), i) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), j) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), a) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), b) )
			  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

		      ham += localham;

		      double Edet = dcopy.Energy(I1, I2, coreE);
		      double ovlpdetcopy = localham/tiajb;
		      double factor = tiajb/(Epsi-Edet) * ovlpdetcopy*scale;
		      if( !schd.davidsonPrecondition) factor *= -(Edet-Epsi);
		      //double factor = tia/(Epsi-Edet);
		      OverlapWithGradient(dcopy, factor, grad);

		    }
		  }
	      }
	  }
      }
  }


  //Double excitation beta beta
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccB(i)) {
	for (int j=i+1; j<Determinant::norbs; j++) 
	  if (d.getoccB(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getoccB(a)) {
		for (int b=a+1; b<Determinant::norbs; b++)
		  if (!d.getoccB(b)) {

		    double tiajb = d.Hij_2ExciteBB(a, i, b, j, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-10) {
		      Determinant dcopy = d;
		      dcopy.setoccB(i, false); dcopy.setoccB(a, true);
		      dcopy.setoccB(j, false); dcopy.setoccB(b, true);

		      localham += tiajb*det.OverlapBB(d, i, j, a, b,  alphainv, betainv)
			*ovlp;

		      for (int n=0; n<cpsArray.size(); n++) 
			if (std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), i) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), j) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), a) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), b) )
			  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

		      ham += localham;

		      double Edet = dcopy.Energy(I1, I2, coreE);
		      double ovlpdetcopy = localham/tiajb;
		      double factor = tiajb/(Epsi-Edet) * ovlpdetcopy*scale;
		      if( !schd.davidsonPrecondition) factor *= -(Edet-Epsi);
		      //double factor = tia/(Epsi-Edet);
		      OverlapWithGradient(dcopy, factor, grad);
		    }
		  }
	      }
	  }
      }
  }


  //Double excitation alpha beta
  {
    for (int i=0; i<Determinant::norbs; i++) 
      if (d.getoccA(i)) {
	for (int j=0; j<Determinant::norbs; j++) 
	  if (d.getoccB(j)) {
	    for (int a=0; a<Determinant::norbs; a++)
	      if (!d.getoccA(a)) {
		for (int b=0; b<Determinant::norbs; b++)
		  if (!d.getoccB(b)) {

		    double tiajb = d.Hij_2ExciteAB(a, i, b, j, I1, I2);
		    double localham = 0.0;
		    if (abs(tiajb) > 1.e-10) {
		      Determinant dcopy = d;
		      dcopy.setoccA(i, false); dcopy.setoccA(a, true);
		      dcopy.setoccB(j, false); dcopy.setoccB(b, true);

		      localham += tiajb*det.OverlapAB(d, i, j, a, b,  alphainv, betainv)
			*ovlp;

		      for (int n=0; n<cpsArray.size(); n++) 
			if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), i) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), j) ||
			    std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), a) ||
			    std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), b) )
			  localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);

		      ham += localham;

		      double Edet = dcopy.Energy(I1, I2, coreE);
		      double ovlpdetcopy = localham/tiajb;
		      double factor = tiajb/(Epsi-Edet) * ovlpdetcopy*scale;
		      if( !schd.davidsonPrecondition) factor *= -(Edet-Epsi);
		      OverlapWithGradient(dcopy, factor, grad);
		    }
		  }
	      }
	  }
      }
  }

}





