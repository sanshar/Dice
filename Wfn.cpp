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

  /*
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
  */

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

  /*
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
  */
}

void CPSSlater::normalizeAllCPS() {
  for (int i=0; i<cpsArray.size(); i++) {
    double norm = 0.;

    for (int a=0; a<cpsArray[i].Variables.size(); a++)
      norm += pow(cpsArray[i].Variables[a], 2);

    for (int a=0; a<cpsArray[i].Variables.size(); a++)
      cpsArray[i].Variables[a] /= sqrt(norm);
  }
}


double CPSSlater::approximateNorm() {
  double norm = 1.0;
  for (int i=0; i<cpsArray.size(); i++) {
    norm *= *max_element(cpsArray[i].Variables.begin(), cpsArray[i].Variables.end());
  }
  return norm;
}

long CPSSlater::getNumVariables() {
  long numVars = 0;
  for (int i=0; i<cpsArray.size(); i++) 
    numVars += cpsArray[i].Variables.size();
  //numVars+=det.norbs*det.nalpha+det.norbs*det.nbeta;
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


void CPSSlater::HamAndOvlp(Walker& walk,
			   double& ovlp, double& ham, oneInt& I1, twoInt& I2, 
			   twoIntHeatBathSHM& I2hb, double& coreE) {

  double TINY = schd.epsilon;
  MatrixXd alphainv = walk.alphainv, betainv = walk.betainv;
  double alphaDet = walk.alphaDet, betaDet = walk.betaDet;
  double detOverlap = alphaDet*betaDet;
  Determinant& d = walk.d;

  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  //noexcitation
  {
    double E0 = d.Energy(I1, I2, coreE);
    ovlp = detOverlap;
    for (int i=0; i<cpsArray.size(); i++)
      ovlp *= cpsArray[i].Overlap(d);
    ham  = E0;

  }


  //Single alpha-beta excitation 
  {
    for (int i=0; i<closed.size(); i++) {
      for (int a=0; a<open.size(); a++) {
	if (closed[i]%2 == open[a]%2) {
	  int I = closed[i]/2, A = open[a]/2;
	  double tia = 0; Determinant dcopy = d;
	  bool Alpha = closed[i]%2 == 0 ? true : false;
	  if (Alpha) tia = d.Hij_1ExciteA(A, I, I1, I2);
	  else tia = d.Hij_1ExciteB(A, I, I1, I2);

	  double localham = 0.0;
	  if (abs(tia) > TINY) {
	    if (Alpha) {dcopy.setoccA(I, false); dcopy.setoccA(A, true);}
	    else  {dcopy.setoccB(I, false); dcopy.setoccB(A, true);}

	    if (Alpha) localham += tia*det.OverlapA(d, I, A,  alphainv, betainv);
	    else localham += tia*det.OverlapB(d, I, A,  alphainv, betainv);

	    for (int n=0; n<cpsArray.size(); n++) 
	      if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), I) ||
		  std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), A) )
		localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);
	    ham += localham;
	    
	  }
	}
      }
    }
  }

  //Double excitation
  {
    int nclosed = closed.size();
    for (int ij=0; ij<nclosed*nclosed; ij++) {
      int i=ij/nclosed, j = ij%nclosed;
      if (i<=j) continue;
      int I = closed[i]/2, J = closed[j]/2;
      int X = max(I, J), Y = min(I, J);

      int pairIndex = X*(X+1)/2+Y;
      size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex]   : I2hb.startingIndicesOppositeSpin[pairIndex];
      size_t end   = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
      float* integrals  = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
      short* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs     : I2hb.oppositeSpinPairs;

      // for all HCI integrals
      for (size_t index=start; index<end; index++) {
	// if we are going below the criterion, break
	if (fabs(integrals[index]) < TINY) break;

	// otherwise: generate the determinant corresponding to the current excitation
	int a = 2* orbIndices[2*index] + closed[i]%2, b= 2*orbIndices[2*index+1]+closed[j]%2;
	if (!(d.getocc(a) || d.getocc(b))) {
	  Determinant dcopy = d;
	  double localham = 0.0;
	  double tiajb = integrals[index];

	  dcopy.setocc(closed[i], false); dcopy.setocc(a, true);
	  dcopy.setocc(closed[j], false); dcopy.setocc(b, true);

	  int A = a/2, B = b/2;
	  int type = 0; //0 = AA, 1 = BB, 2 = AB, 3 = BA
	  if (closed[i]%2==closed[j]%2 && closed[i]%2 == 0) 
	    localham += tiajb*det.OverlapAA(d, I, J, A, B,  alphainv, betainv, false);
	  else if (closed[i]%2==closed[j]%2 && closed[i]%2 == 1) 
	    localham += tiajb*det.OverlapBB(d, I, J, A, B,  alphainv, betainv, false);
	  else if (closed[i]%2!=closed[j]%2 && closed[i]%2 == 0)
	    localham += tiajb*det.OverlapAB(d, I, J, A, B,  alphainv, betainv, false);
	  else
	    localham += tiajb*det.OverlapAB(d, J, I, B, A,  alphainv, betainv, false);

	  for (int n=0; n<cpsArray.size(); n++) 
	    if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), I) ||
		std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), J) ||
		std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), A) ||
		std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), B) )
	      localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);
	  
	  ham += localham;
	}
      }
    }

  }
}


//<psi_t| (H-E0) |D>
void CPSSlater::HamAndOvlpGradient(Walker& walk,
				   double& ovlp, double& ham, VectorXd& grad, double& scale,
				   double& Epsi, oneInt& I1, twoInt& I2, 
				   twoIntHeatBathSHM& I2hb, double& coreE) {

  double TINY = schd.epsilon;
  MatrixXd alphainv = walk.alphainv, betainv = walk.betainv;
  double alphaDet = walk.alphaDet, betaDet = walk.betaDet;
  double detOverlap = alphaDet*betaDet;
  Determinant& d = walk.d;

  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  //noexcitation
  {
    double E0 = d.Energy(I1, I2, coreE);
    ovlp = detOverlap;
    for (int i=0; i<cpsArray.size(); i++)
      ovlp *= cpsArray[i].Overlap(d);
    ham  = E0;

    double factor = schd.davidsonPrecondition ? ovlp*scale : (E0-Epsi)*scale;    
    OverlapWithGradient(d, factor, grad);
  }


  //Single alpha-beta excitation 
  {
    for (int i=0; i<closed.size(); i++) {
      for (int a=0; a<open.size(); a++) {
	if (closed[i]%2 == open[a]%2) {
	  int I = closed[i]/2, A = open[a]/2;
	  double tia = 0; Determinant dcopy = d;
	  bool Alpha = closed[i]%2 == 0 ? true : false;
	  if (Alpha) tia = d.Hij_1ExciteA(A, I, I1, I2);
	  else tia = d.Hij_1ExciteB(A, I, I1, I2);

	  double localham = 0.0;
	  if (abs(tia) > TINY) {
	    if (Alpha) {dcopy.setoccA(I, false); dcopy.setoccA(A, true);}
	    else  {dcopy.setoccB(I, false); dcopy.setoccB(A, true);}

	    if (Alpha) localham += tia*det.OverlapA(d, I, A,  alphainv, betainv);
	    else localham += tia*det.OverlapB(d, I, A,  alphainv, betainv);

	    for (int n=0; n<cpsArray.size(); n++) 
	      if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), I) ||
		  std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), A) )
		localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);
	    ham += localham;
	    
	    double ovlpdetcopy = localham/tia;
	    double factor = tia * ovlpdetcopy*scale;
	    OverlapWithGradient(dcopy, factor, grad);
	    
	  }
	}
      }
    }
  }

  //Double excitation
  {
    int nclosed = closed.size();
    for (int ij=0; ij<nclosed*nclosed; ij++) {
      int i=ij/nclosed, j = ij%nclosed;
      if (i<=j) continue;
      int I = closed[i]/2, J = closed[j]/2;
      int X = max(I, J), Y = min(I, J);

      int pairIndex = X*(X+1)/2+Y;
      size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex]   : I2hb.startingIndicesOppositeSpin[pairIndex];
      size_t end   = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
      float* integrals  = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
      short* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs     : I2hb.oppositeSpinPairs;

      // for all HCI integrals
      for (size_t index=start; index<end; index++) {
	// if we are going below the criterion, break
	if (fabs(integrals[index]) < TINY) break;

	// otherwise: generate the determinant corresponding to the current excitation
	int a = 2* orbIndices[2*index] + closed[i]%2, b= 2*orbIndices[2*index+1]+closed[j]%2;
	if (!(d.getocc(a) || d.getocc(b))) {
	  Determinant dcopy = d;
	  double localham = 0.0;
	  double tiajb = integrals[index];

	  dcopy.setocc(closed[i], false); dcopy.setocc(a, true);
	  dcopy.setocc(closed[j], false); dcopy.setocc(b, true);

	  int A = a/2, B = b/2;
	  int type = 0; //0 = AA, 1 = BB, 2 = AB, 3 = BA
	  if (closed[i]%2==closed[j]%2 && closed[i]%2 == 0) 
	    localham += tiajb*det.OverlapAA(d, I, J, A, B,  alphainv, betainv, false);
	  else if (closed[i]%2==closed[j]%2 && closed[i]%2 == 1) 
	    localham += tiajb*det.OverlapBB(d, I, J, A, B,  alphainv, betainv, false);
	  else if (closed[i]%2!=closed[j]%2 && closed[i]%2 == 0)
	    localham += tiajb*det.OverlapAB(d, I, J, A, B,  alphainv, betainv, false);
	  else
	    localham += tiajb*det.OverlapAB(d, J, I, B, A,  alphainv, betainv, false);

	  for (int n=0; n<cpsArray.size(); n++) 
	    if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), I) ||
		std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), J) ||
		std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), A) ||
		std::binary_search(cpsArray[n].bsites.begin(), cpsArray[n].bsites.end(), B) )
	      localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);
	  
	  ham += localham;
	  
	  double ovlpdetcopy = localham/tiajb;
	  double factor = tiajb * ovlpdetcopy*scale;
	  OverlapWithGradient(dcopy, factor, grad);
	}
      }
    }

  }
}





