#include "Walker.h"
#include "Wfn.h"
#include "integral.h"
#include "global.h"
#include "input.h"
using namespace Eigen;

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
