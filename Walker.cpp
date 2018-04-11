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
#include "Walker.h"
#include "Wfn.h"
#include "integral.h"
using namespace Eigen;


bool Walker::makeMove(CPSSlater& w) {
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  int norbs = MoDeterminant::norbs,
    nalpha = MoDeterminant::nalpha, 
    nbeta  = MoDeterminant::nbeta; 
  
  //pick a random occupied orbital
  int i = floor( random()*(nalpha+nbeta) );
  if (i < nalpha) {
    int a = floor(random()* (norbs-nalpha) );
    double detfactor = getDetFactorA(i, a, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;
    if ( pow(detfactor, 2) > random() ) {
      updateA(i, a, w);
      return true;
    }
    
  }
  else {
    i = i - nalpha;
    int a = floor( random()*(norbs-nbeta));
    double detfactor = getDetFactorB(i, a, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;    
    if ( pow(detfactor, 2) > random() ) {
      updateB(i, a, w);
      return true;
    }
    
  }
  
  return false;
}


bool Walker::makeCleverMove(CPSSlater& w, oneInt& I1) {
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  //double pickOpenElec = 1.0, pickClosedElec = 5.0;
  //double pickOpenHole = 5.0, pickClosedHole = 1.0;
  double pickOpenElec = 1.0, pickClosedElec = 2.0;
  double pickOpenHole = 2.0, pickClosedHole = 1.0;

  double epsilon = 1.e-4;
  int norbs = MoDeterminant::norbs,
    nalpha = MoDeterminant::nalpha, 
    nbeta  = MoDeterminant::nbeta; 

  

  //pick a random occupied orbital
  int aorb = floor( random()*(nalpha+nbeta) );
  if (aorb < nalpha) {
    vector<int> openE, closedE, openH, closedH;
    //d.getOpenClosedElecHoleAlpha(openE, closedE, openH, closedH);
    
    double probForward=1.;

    //pickElectron
    double totalCountE = (pickOpenElec*openE.size() 
			  + pickClosedElec*closedE.size());
    double picki = random()*totalCountE;
    //cout << totalCountE<<"  "<<picki<<"  "<<openE.size()<<"  "<<closedE.size()<<endl;
    int i=0, occi=0;
    for (int n=0; n<norbs; n++) {
      if (d.getoccA(n) && d.getoccB(n) && pickClosedElec > picki) {
	probForward *= pickClosedElec/totalCountE;
	occi = n;
	break;
      }
      else if (d.getoccA(n) && !d.getoccB(n) && pickOpenElec > picki) {
	probForward *= pickOpenElec/totalCountE;
	occi = n;
	break;
      }
      else if (d.getoccA(n) && d.getoccB(n) && pickClosedElec < picki) {
	picki -= pickClosedElec;
	i++;
      }
      else if (d.getoccA(n) && !d.getoccB(n) && pickOpenElec < picki) {
	picki -= pickOpenElec;
	i++;
      }
    }

    //pickHole
    double totalCountH = (pickOpenHole*openH.size() 
			  + pickClosedHole*closedH.size());
    double picka = random()*totalCountH;
    //cout << totalCountH<<"  "<<picka<<"  "<<openH.size()<<"  "<<closedH.size()<<endl;

    int a=0, occa=0;
    for (int n=0; n<norbs; n++) {
      if (!d.getoccA(n) && d.getoccB(n) && pickClosedHole > picka) {
	probForward *= pickClosedHole/totalCountH;
	occa = n;
	break;
      }
      else if (!d.getoccA(n) && !d.getoccB(n) && pickOpenHole > picka) {
	probForward *= pickOpenHole/totalCountH;
	occa = n;
	break;
      }
      else if (!d.getoccA(n) && d.getoccB(n) && pickClosedElec < picka) {
	picka -= pickClosedHole;
	a++;
      }
      else if (!d.getoccA(n) && !d.getoccB(n) && pickOpenHole < picka) {
	picka -= pickOpenHole;
	a++;
      }
    }

    double probReverse = 1.0;
    {
      double pickH , pickE;
      if (find(openE.begin(), openE.end(), occi) != openE.end()) {
	totalCountE -= pickOpenElec;
	totalCountH += pickOpenHole;
	pickH = pickOpenHole;
      }
      else {
	totalCountE -= pickClosedElec;
	totalCountH += pickClosedHole;
	pickH = pickClosedHole;
      }
      if (find(openH.begin(), openH.end(), occa) != openH.end()) {
	totalCountE += pickOpenElec;
	totalCountH -= pickOpenHole;
	pickE = pickOpenElec;
      }
      else {
	totalCountE += pickClosedElec;
	totalCountH -= pickClosedHole;
	pickE = pickClosedElec;
      }
      probReverse = (pickH/totalCountH)*(pickE/totalCountE);
    }

    double detfactor = pow(getDetFactorA(i, a, w),2) * probReverse/probForward;
    
    Determinant dcopy = d; dcopy.setoccA(occi,false); dcopy.setoccA(occa, true); 
    //cout << i<<"  "<<occi<<"  "<<a<<" "<<occa<<endl;
    //cout << pow(getDetFactorA(i, a, w), 2)<<"  "<<probReverse/probForward<<"  "<<d<<"   "<<dcopy<<endl;
    if ( detfactor > random() ) {
      updateA(i, a, w);
      return true;
    }

  }
  else {
    /*
    {
      int i = floor( random()*nbeta);//**********
      int a = floor( random()*(norbs-nbeta));
      double detfactor = getDetFactorB(i, a, w);
      
      if ( pow(detfactor, 2) > random() ) {
	updateB(i, a, w);
	return true;
      }
      else return false;
    }
    */
    vector<int> openE, closedE, openH, closedH;
    //d.getOpenClosedElecHoleBeta(openE, closedE, openH, closedH);
    
    double probForward=1.;

    //pickElectron
    double totalCountE = (pickOpenElec*openE.size() 
			  + pickClosedElec*closedE.size());
    double picki = random()*totalCountE;
    //cout << openE.size()<<"  "<<closedE.size()<<endl;
    //cout << totalCountE<<"  "<<picki<<endl;
    int i=0, occi=0;
    for (int n=0; n<norbs; n++) {
      if (d.getoccB(n) && d.getoccA(n) && pickClosedElec > picki) {
	probForward *= pickClosedElec/totalCountE;
	occi = n;
	break;
      }
      else if (d.getoccB(n) && !d.getoccA(n) && pickOpenElec > picki) {
	probForward *= pickOpenElec/totalCountE;
	occi = n;
	break;
      }
      else if (d.getoccB(n) && d.getoccA(n) && pickClosedElec < picki) {
	picki -= pickClosedElec;
	//cout << picki<<"  "<<n<<"  "<<pickClosedElec<<endl;
	i++;
      }
      else if (d.getoccB(n) && !d.getoccA(n) && pickOpenElec < picki) {
	picki -= pickOpenElec;
	//cout << picki<<"  "<<n<<"  "<<pickOpenElec<<endl;
	i++;
      }
    }

    //pickHole
    double totalCountH = (pickOpenHole*openH.size() 
			  + pickClosedHole*closedH.size());
    double picka = random()*totalCountH;

    int a=0, occa=0;
    for (int n=0; n<norbs; n++) {
      if (!d.getoccB(n) && d.getoccA(n) && pickClosedHole > picka) {
	probForward *= pickClosedHole/totalCountH;
	occa = n;
	break;
      }
      else if (!d.getoccB(n) && !d.getoccA(n) && pickOpenHole > picka) {
	probForward *= pickOpenHole/totalCountH;
	occa = n;
	break;
      }
      else if (!d.getoccB(n) && d.getoccA(n) && pickClosedElec < picka) {
	picka -= pickClosedHole;
	a++;
      }
      else if (!d.getoccB(n) && !d.getoccA(n) && pickOpenHole < picka) {
	picka -= pickOpenHole;
	a++;
      }
    }

    double probReverse = 1.0;
    {
      double pickH , pickE;
      if (find(openE.begin(), openE.end(), occi) != openE.end()) {
	totalCountE -= pickOpenElec;
	totalCountH += pickOpenHole;
	pickH = pickOpenHole;
      }
      else {
	totalCountE -= pickClosedElec;
	totalCountH += pickClosedHole;
	pickH = pickClosedHole;
      }
      if (find(openH.begin(), openH.end(), occa) != openH.end()) {
	totalCountE += pickOpenElec;
	totalCountH -= pickOpenHole;
	pickE = pickOpenElec;
      }
      else {
	totalCountE += pickClosedElec;
	totalCountH -= pickClosedHole;
	pickE = pickClosedElec;
      }
      probReverse = (pickH/totalCountH)*(pickE/totalCountE);
    }

    double detfactor = pow(getDetFactorB(i, a, w),2) * probReverse/probForward;

    if ( detfactor > random() ) {
      updateB(i, a, w);
      return true;
    }
    
  }
  
  return false;
}

void Walker::initUsingWave(Wfn& w, bool check) {
  MatrixXd alpha, beta;
  w.getDetMatrix(d, alpha, beta);

  Eigen::FullPivLU<MatrixXd> lua(alpha);
  if (lua.isInvertible() || !check) {
    alphainv = lua.inverse();
    alphaDet = lua.determinant();
  }
  else {
    cout << "overlap with alpha determinant "<< d <<" no invertible"<<endl;
    cout << "rank of the matrix: "<<lua.rank()<<endl;
    EigenSolver<MatrixXx> eigensolver(alpha);
    cout << eigensolver.eigenvalues()<<endl;
    exit(0);
  }

  Eigen::FullPivLU<MatrixXd> lub(beta);
  if (lub.isInvertible() || !check) {
    betainv = lub.inverse();
    betaDet = lub.determinant();
  }
  else {
    cout << "overlap with beta determinant "<< d <<" no invertible"<<endl;
    cout << "rank of the matrix: "<<lub.rank()<<endl;
    EigenSolver<MatrixXx> eigensolver(beta);
    cout << eigensolver.eigenvalues()<<endl;
    exit(0);
  }

}

double Walker::getDetFactorA(int i, int a, CPSSlater& w) {
  int occi = -1, occa=-1;//d.getNalphaBefore(i);
  for (int n=0; n<Determinant::norbs; n++) {
    if (!d.getoccA(n) && (n-d.getNalphaBefore(n)) == a) {
      occa = n;
      break;
    }
  }
  for (int n=0; n<Determinant::norbs; n++) {
    if (d.getoccA(n) && (d.getNalphaBefore(n)) == i) {
      occi = n;
      break;
    }
  }
  if (occi == -1) occi = Determinant::norbs-1;
  if (occa == -1) occa = Determinant::norbs-1;

  double p=1.;
  d.parityA(occa, occi, p);

  Determinant dcopy = d;
  dcopy.setoccA(occi, false); dcopy.setoccA(occa, true);
  double cpsFactor = 1.0;
  for (int n=0; n<w.cpsArray.size(); n++) 
    if (std::binary_search(w.cpsArray[n].asites.begin(), w.cpsArray[n].asites.end(), occi) ||
	std::binary_search(w.cpsArray[n].asites.begin(), w.cpsArray[n].asites.end(), occa) )
      cpsFactor *= w.cpsArray[n].Overlap(dcopy)/w.cpsArray[n].Overlap(d);

  return p*cpsFactor*(1+(w.det.AlphaOrbitals.row(occa)-
		       w.det.AlphaOrbitals.row(occi))*alphainv.col(i));
}

void Walker::updateA(int i, int a, CPSSlater& w) {
  double p = 1.0;
  int occi = -1, occa=-1;//d.getNalphaBefore(i);
  for (int n=0; n<Determinant::norbs; n++) {
    if (!d.getoccA(n) && (n-d.getNalphaBefore(n)) == a) {
      occa = n;
      break;
    }
  }
  for (int n=0; n<Determinant::norbs; n++) {
    if (d.getoccA(n) && (d.getNalphaBefore(n)) == i) {
      occi = n;
      break;
    }
  }
  if (occi == -1) occi = Determinant::norbs-1;
  if (occa == -1) occa = Determinant::norbs-1;

  d.parityA(occa, occi, p);

  double alphaDetFactor = (1+(w.det.AlphaOrbitals.row(occa)-
			      w.det.AlphaOrbitals.row(occi))*alphainv.col(i));
  alphaDet *= alphaDetFactor;
  MatrixXd vtAinv = (w.det.AlphaOrbitals.row(occa)-
		     w.det.AlphaOrbitals.row(occi))*alphainv;
  
  MatrixXd alphainvWrongOrder =  alphainv - (alphainv.col(i) * vtAinv)/alphaDetFactor;

  std::vector<int> alphaOrbitalOrder, betaOrbitalOrder;
  d.getAlphaBeta(alphaOrbitalOrder, betaOrbitalOrder);

  std::vector<int> order(alphaOrbitalOrder.size());
  alphaOrbitalOrder[i] = occa;
  for (int i=0; i<order.size(); i++) order[i] = i;
  std::sort(order.begin(), order.end(), 
	    [&alphaOrbitalOrder](size_t i1, size_t i2) 
	    {return alphaOrbitalOrder[i1] < alphaOrbitalOrder[i2];});

  for (int i=0; i<order.size(); i++) {
    alphainv.col(i) = alphainvWrongOrder.col(order[i]);
  }

  alphaDet *= p;
  d.setoccA(occi, false); d.setoccA(occa, true);
}

double Walker::getDetFactorB(int i, int a, CPSSlater& w) {
  int occi = -1, occa=-1;//d.getNalphaBefore(i);
  for (int n=0; n<Determinant::norbs; n++) {
    if (!d.getoccB(n) && (n-d.getNbetaBefore(n)) == a) {
      occa = n;
      break;
    }
  }
  for (int n=0; n<Determinant::norbs; n++) {
    if (d.getoccB(n) && (d.getNbetaBefore(n)) == i) {
      occi = n;
      break;
    }
  }
  if (occi == -1) occi = Determinant::norbs-1;
  if (occa == -1) occa = Determinant::norbs-1;

  double p=1.;
  d.parityB(occa, occi, p);

  Determinant dcopy = d;
  dcopy.setoccB(occi, false); dcopy.setoccB(occa, true);
  double cpsFactor = 1.0;
  for (int n=0; n<w.cpsArray.size(); n++) 
    if (std::binary_search(w.cpsArray[n].bsites.begin(), w.cpsArray[n].bsites.end(), occi) ||
	std::binary_search(w.cpsArray[n].bsites.begin(), w.cpsArray[n].bsites.end(), occa) )
      cpsFactor *= w.cpsArray[n].Overlap(dcopy)/w.cpsArray[n].Overlap(d);

  return p*cpsFactor*(1+(w.det.BetaOrbitals.row(occa)-
		       w.det.BetaOrbitals.row(occi))*betainv.col(i));

}

void Walker::updateB(int i, int a, CPSSlater& w) {
  double p = 1.0;
  int occi = -1, occa=-1;//d.getNalphaBefore(i);
  for (int n=0; n<Determinant::norbs; n++) {
    if (!d.getoccB(n) && (n-d.getNbetaBefore(n)) == a) {
      occa = n;
      break;
    }
  }
  for (int n=0; n<Determinant::norbs; n++) {
    if (d.getoccB(n) && (d.getNbetaBefore(n)) == i) {
      occi = n;
      break;
    }
  }
  if (occi == -1) occi = Determinant::norbs-1;
  if (occa == -1) occa = Determinant::norbs-1;

  d.parityB(occa, occi, p);

  double betaDetFactor = (1+(w.det.BetaOrbitals.row(occa)-
			      w.det.BetaOrbitals.row(occi))*betainv.col(i));
  betaDet *= betaDetFactor;
  MatrixXd vtAinv = (w.det.BetaOrbitals.row(occa)-
		     w.det.BetaOrbitals.row(occi))*betainv;
  
  MatrixXd betainvWrongOrder =  betainv - (betainv.col(i) * vtAinv)/betaDetFactor;

  std::vector<int> alphaOrbitalOrder, betaOrbitalOrder;
  d.getAlphaBeta(alphaOrbitalOrder, betaOrbitalOrder);
  std::vector<int> order(betaOrbitalOrder.size());
  betaOrbitalOrder[i] = occa;
  for (int i=0; i<order.size(); i++) order[i] = i;
  std::sort(order.begin(), order.end(), 
	    [&betaOrbitalOrder](size_t i1, size_t i2) 
	    {return betaOrbitalOrder[i1] < betaOrbitalOrder[i2];});

  for (int i=0; i<order.size(); i++) {
    betainv.col(i) = betainvWrongOrder.col(order[i]);
  }

  betaDet *= p;
  d.setoccB(occi, false); d.setoccB(occa, true);
}
