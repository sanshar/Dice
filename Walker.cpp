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
#include "global.h"
#include "input.h"
using namespace Eigen;

int getAbsoluteClosedIndexA(int i, Determinant& d){
  int occi = -1, occa=-1;//d.getNalphaBefore(i);
  for (int n=0; n<Determinant::norbs; n++) {
    if (d.getoccA(n) && (d.getNalphaBefore(n)) == i) {
      occi = n;
      break;
    }
  }
  if (occi == -1) occi = Determinant::norbs-1;
  return occi;
}

int getAbsoluteOpenIndexA(int a, Determinant& d){
  int occi = -1, occa=-1;//d.getNalphaBefore(i);
  for (int n=0; n<Determinant::norbs; n++) {
    if (!d.getoccA(n) && (n-d.getNalphaBefore(n)) == a) {
      occa = n;
      break;
    }
  }
  if (occa == -1) occa = Determinant::norbs-1;
  return occa;
}

int getAbsoluteClosedIndexB(int i, Determinant& d){
  int occi = -1, occa=-1;//d.getNalphaBefore(i);
  for (int n=0; n<Determinant::norbs; n++) {
    if (d.getoccB(n) && (d.getNbetaBefore(n)) == i) {
      occi = n;
      break;
    }
  }
  if (occi == -1) occi = Determinant::norbs-1;
  return occi;
}

int getAbsoluteOpenIndexB(int a, Determinant& d){
  int occi = -1, occa=-1;//d.getNalphaBefore(i);
  for (int n=0; n<Determinant::norbs; n++) {
    if (!d.getoccB(n) && (n-d.getNbetaBefore(n)) == a) {
      occa = n;
      break;
    }
  }
  if (occa == -1) occa = Determinant::norbs-1;
  return occa;
}

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
    int I = getAbsoluteClosedIndexA(i, d);
    int A = getAbsoluteOpenIndexA(a, d);
    double detfactor = getDetFactorA(I, A, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;
    if ( pow(detfactor, 2) > random() ) {
      updateA(I, A, w);
      return true;
    }
    
  }
  else {
    i = i - nalpha;
    int a = floor( random()*(norbs-nbeta));
    int I = getAbsoluteClosedIndexB(i, d);
    int A = getAbsoluteOpenIndexB(a, d);
    double detfactor = getDetFactorB(I, A, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;    
    if ( pow(detfactor, 2) > random() ) {
      updateB(I, A, w);
      return true;
    }
    
  }
  
  return false;
}


bool Walker::makeMovePropPsi(CPSSlater& w) {
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  int norbs = MoDeterminant::norbs,
    nalpha = MoDeterminant::nalpha, 
    nbeta  = MoDeterminant::nbeta; 
  
  //pick a random occupied orbital
  int i = floor( random()*(nalpha+nbeta) );
  if (i < nalpha) {
    int a = floor(random()* (norbs-nalpha) );
    int I = getAbsoluteClosedIndexA(i, d);
    int A = getAbsoluteOpenIndexA(a, d);
    double detfactor = getDetFactorA(I, A, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;
    if ( abs(detfactor) > random() ) {
      updateA(I, A, w);
      return true;
    }
    
  }
  else {
    i = i - nalpha;
    int a = floor( random()*(norbs-nbeta));
    int I = getAbsoluteClosedIndexB(i, d);
    int A = getAbsoluteOpenIndexB(a, d);
    double detfactor = getDetFactorB(I, A, w);
    //cout << i<<"   "<<a<<"   "<<detfactor<<endl;    
    if ( abs(detfactor) > random() ) {
      updateB(I, A, w);
      return true;
    }
    
  }
  
  return false;
}




void Walker::genAllMoves(CPSSlater& w, vector<Determinant>& dout, 
			  vector<double>& prob, vector<size_t>& alphaExcitation,
			  vector<size_t>& betaExcitation) {

  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);

  //generate all single excitation
  for (int i=0; i<closed.size(); i++) {
    for (int a=0; a<open.size(); a++) {
      if (closed[i]%2 == open[a]%2) {
	if (closed[i]%2 == 0) {
	  Determinant dcopy = d;
	  dcopy.setoccA(closed[i]/2, false); dcopy.setoccA(open[a]/2, true);
	  dout.push_back(dcopy);
	  prob.push_back(getDetFactorA(closed[i]/2, open[a]/2, w));
	  alphaExcitation.push_back(closed[i]/2*Determinant::norbs+open[a]/2);
	  betaExcitation.push_back(0);
	  //cout << " alpha "<<*alphaExcitation.rbegin()<<"  "<<closed[i]<<"  "<<open[a]<<endl;
	}
	else {
	  Determinant dcopy = d;
	  dcopy.setoccB(closed[i]/2, false); dcopy.setoccB(open[a]/2, true);
	  dout.push_back(dcopy);
	  prob.push_back(getDetFactorB(closed[i]/2, open[a]/2, w));
	  alphaExcitation.push_back(0);
	  betaExcitation.push_back(closed[i]/2*Determinant::norbs+open[a]/2);	
	  //cout << " beta "<<*betaExcitation.rbegin()<<"  "<<closed[i]<<"  "<<open[a]<<endl;
	}
      }
    }
  }


  //alpha-beta electron exchanges
  for (int i=0; i<closed.size(); i++) {
    if (closed[i]%2 == 0 && !d.getocc(closed[i]+1)) {//alpha orbital and single alpha occupation 
      for (int j=0; j<closed.size(); j++) {
	if (closed[j]%2 == 1 && !d.getocc(closed[j]-1)) {//beta orbital and single beta occupation 
	  Determinant dcopy = d;
	  dcopy.setoccA(closed[i]/2, false); dcopy.setoccB(closed[i]/2, true);
	  dcopy.setoccA(closed[j]/2, true) ; dcopy.setoccB(closed[j]/2, false);
	  prob.push_back(getDetFactorA(closed[i]/2, closed[j]/2,w)
			 *getDetFactorB(closed[j]/2, closed[i]/2,w));
	  alphaExcitation.push_back(closed[i]/2*Determinant::norbs+closed[j]/2);
	  betaExcitation.push_back(closed[j]/2*Determinant::norbs+closed[i]/2);
	  //cout << " alpha/beta "<<*alphaExcitation.rbegin()<<"  "<<*betaExcitation.rbegin()<<"  "<<closed[i]<<"  "<<closed[j]<<endl;
	}
      }
    }
  }

}


void Walker::genAllMoves2(CPSSlater& w, vector<Determinant>& dout, 
			  vector<double>& prob, vector<size_t>& alphaExcitation,
			  vector<size_t>& betaExcitation) {
  
  vector<int> closed;
  vector<int> open;
  d.getOpenClosed(open, closed);


  //alpha-beta electron exchanges
  if (schd.doubleProbability > 1.e-10) {
    for (int i=0; i<closed.size(); i++) {
      if (closed[i]%2 == 0 && !d.getocc(closed[i]+1)) {//alpha orbital and single alpha occupation 
	for (int j=0; j<closed.size(); j++) {
	  if (closed[j]%2 == 1 && !d.getocc(closed[j]-1)) {//beta orbital and single beta occupation 
	    Determinant dcopy = d;
	    dcopy.setoccA(closed[i]/2, false); dcopy.setoccB(closed[i]/2, true);
	    dcopy.setoccA(closed[j]/2, true) ; dcopy.setoccB(closed[j]/2, false);
	    prob.push_back(schd.doubleProbability);
	    alphaExcitation.push_back(closed[i]/2*Determinant::norbs+closed[j]/2);
	    betaExcitation.push_back(closed[j]/2*Determinant::norbs+closed[i]/2);
	    //cout << " alpha/beta "<<*alphaExcitation.rbegin()<<"  "<<*betaExcitation.rbegin()<<"  "<<closed[i]<<"  "<<closed[j]<<endl;
	  }
	}
      }
    }
  }

  if (prob.size() == 0 || schd.singleProbability > 1.e-10) {
    //generate all single excitation
    for (int i=0; i<closed.size(); i++) {
      for (int a=0; a<open.size(); a++) {
	if (closed[i]%2 == open[a]%2) {
	  if (closed[i]%2 == 0) {
	    Determinant dcopy = d;
	    dcopy.setoccA(closed[i]/2, false); dcopy.setoccA(open[a]/2, true);
	    dout.push_back(dcopy);
	    //if (dcopy.getocc(open[a]+1)) 
	      prob.push_back(schd.singleProbability);
	      //else
	      //prob.push_back(10.*schd.singleProbability);

	    alphaExcitation.push_back(closed[i]/2*Determinant::norbs+open[a]/2);
	    betaExcitation.push_back(0);
	    //cout << " alpha "<<*alphaExcitation.rbegin()<<"  "<<closed[i]<<"  "<<open[a]<<endl;
	  }
	  else {
	    Determinant dcopy = d;
	    dcopy.setoccB(closed[i]/2, false); dcopy.setoccB(open[a]/2, true);
	    dout.push_back(dcopy);
	    //if (dcopy.getocc(open[a]-1)) 
	      prob.push_back(schd.singleProbability);
	      //else
	      //prob.push_back(10.*schd.singleProbability);

	    alphaExcitation.push_back(0);
	    betaExcitation.push_back(closed[i]/2*Determinant::norbs+open[a]/2);	
	    //cout << " beta "<<*betaExcitation.rbegin()<<"  "<<closed[i]<<"  "<<open[a]<<endl;
	  }
	}
      }
    }
  }


}


bool Walker::makeCleverMove(CPSSlater& w) {
  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  double probForward, ovlpForward=1.0;
  Walker wcopy = *this;


  {
    vector<Determinant> dout;
    vector<double> prob;
    vector<size_t> alphaExcitation, betaExcitation;
    genAllMoves2(w, dout, prob, alphaExcitation, betaExcitation);
    
    //pick one determinant at random
    double cumulForward = 0;
    vector<double> cumulative(prob.size(), 0);
    double prev = 0.;
    for (int i=0; i<prob.size(); i++) {
      cumulForward += prob[i];
      cumulative[i] = prev + prob[i];
      prev = cumulative[i];
    }
    
    double selectForward = random()*cumulForward;
    //cout << selectForward/cumulForward<<endl;

    int detIndex= std::lower_bound(cumulative.begin(), cumulative.end(), selectForward)-cumulative.begin();
    
    probForward = prob[detIndex]/cumulForward;

    if (alphaExcitation[detIndex] != 0) {
      int I = alphaExcitation[detIndex]/Determinant::norbs;
      int A = alphaExcitation[detIndex]-I*Determinant::norbs;
      //cout <<" a "<<I<<"  "<<A<<"  "<<wcopy.d<<"  "<<alphaExcitation[detIndex]<<endl;
      ovlpForward *= getDetFactorA(I, A, w);
      wcopy.updateA(I, A, w);
    }
    if (betaExcitation[detIndex] != 0) {
      int I = betaExcitation[detIndex]/Determinant::norbs;
      int A = betaExcitation[detIndex]-I*Determinant::norbs;
      //cout <<" b "<<I<<"  "<<A<<endl;
      ovlpForward *= getDetFactorB(I, A, w);
      wcopy.updateB(I, A, w);
    }
    
  }

  double probBackward, ovlpBackward;
  {
    vector<Determinant> dout;
    vector<double> prob;
    vector<size_t> alphaExcitation, betaExcitation;
    wcopy.genAllMoves2(w, dout, prob, alphaExcitation, betaExcitation);
    
    //pick one determinant at random
    double cumulBackward = 0;
    int index = -1;
    for (int i=0; i<prob.size(); i++) {
      cumulBackward += prob[i];
      if (dout[i] == this->d) 
	index = i;
    }
    //cout << "backward    "<<prob[index]<<"  "<<dout[index]<<"  "<<cumulBackward<<endl;
    probBackward = prob[index]/cumulBackward;
  }

  //cout << ovlpForward<<"  "<<probBackward<<"  "<<probForward<<endl;

  double acceptance = pow(ovlpForward,2)*probBackward/probForward;

  if (acceptance > random()) {
    *this = wcopy;
    return true;
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
  int occi = i, occa=a;
  i = d.getNalphaBefore(occi);

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

  int occi = i, occa = a;
  i = d.getNalphaBefore(occi);
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
  int occi = i, occa=a;
  i = d.getNbetaBefore(occi);

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
  int occi = i, occa=a;
  i = d.getNbetaBefore(occi);

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
