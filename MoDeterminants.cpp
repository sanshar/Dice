#include "MoDeterminants.h"
#include "integral.h"
#include "Determinants.h"

using namespace Eigen;
using namespace std;

double MoDeterminant::Overlap(Determinant& d) {

  std::vector<int> alpha, beta;
  d.getAlphaBeta(alpha, beta);
  return Overlap(alpha, beta);
}

void MoDeterminant::HamAndOvlp(Determinant& d,
			       double& ovlp, double& ham,
			       oneInt& I1, twoInt& I2, double& coreE) {
  std::vector<int> alpha, beta;
  d.getAlphaBeta(alpha, beta);
  HamAndOvlp(alpha, beta, ovlp, ham, I1, I2, coreE);

}

double MoDeterminant::OverlapA(Determinant& d, int i, int a,
			      Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
			      bool doparity) {
  double p = 1.0;
  if (doparity) d.parityA(a, i, p);
  int occ = d.getNalphaBefore(i);
  return p*AlphaOrbitals.row(a)*alphainv.col(occ);
}

double MoDeterminant::OverlapB(Determinant& d, int i, int a,
			      Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
			      bool doparity) {
  double p = 1.0;
  if (doparity) d.parityB(a, i, p);
  int occ = d.getNbetaBefore(i);
  return p*BetaOrbitals.row(a)*betainv.col(occ);
}


double MoDeterminant::OverlapAA(Determinant& d, int i, int j, int a, int b,
				Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
				bool doparity) {

  double p = 1.0;

  double factor1 = 1.0, factor2 = 1.0, factor3 = 1.0, factor4 = 1.0;

  factor1 = OverlapA(d, i, a, alphainv, betainv, false);
  factor2 = OverlapA(d, j, b, alphainv, betainv, false);
  factor3 = OverlapA(d, i, b, alphainv, betainv, false);
  factor4 = OverlapA(d, j, a, alphainv, betainv, false);

  if (doparity) {
    d.parityA(a, i, p); d.setoccA(a, true); d.setoccA(i, false);
    d.parityA(b, j, p); d.setoccA(a, false); d.setoccA(i, true);
  }
  return p*(factor1*factor2 - factor3*factor4);
}

double MoDeterminant::OverlapAAA(Determinant& d, int i, int j, int k,
				 int a, int b, int c,
				 Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
				 bool doparity) {

  vector<int> occ(3,0); occ[0] = i; occ[1] = j; occ[2] = k;
  vector<int> vir(3,0); vir[0] = a; vir[1] = b; vir[2] = c;

  Matrix3f m ;

  for (int x=0; x<3; x++)
    for (int y=0; y<3; y++)
      m(x,y) = OverlapA(d, occ[x], vir[y], alphainv, betainv, false);

  if (doparity) {
    cout << "not supported, SORRY!!"<<endl;
    exit(0);
  }
  return m.determinant();
}

double MoDeterminant::OverlapAAAA(Determinant& d, int i, int j, int k, int l,
				  int a, int b, int c, int e,
				  Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
				  bool doparity) {

  vector<int> occ(4,0); occ[0] = i; occ[1] = j; occ[2] = k; occ[3] = l;
  vector<int> vir(4,0); vir[0] = a; vir[1] = b; vir[2] = c; vir[3] = e;

  Matrix4f m ;

  for (int x=0; x<4; x++)
    for (int y=0; y<4; y++)
      m(x,y) = OverlapA(d, occ[x], vir[y], alphainv, betainv, false);

  if (doparity) {
    cout << "not supported, SORRY!!"<<endl;
    exit(0);
  }
  return m.determinant();
}

double MoDeterminant::OverlapBBB(Determinant& d, int i, int j, int k,
				 int a, int b, int c,
				 Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
				 bool doparity) {

  vector<int> occ(3,0); occ[0] = i; occ[1] = j; occ[2] = k;
  vector<int> vir(3,0); vir[0] = a; vir[1] = b; vir[2] = c;

  Matrix3f m ;

  for (int x=0; x<3; x++)
    for (int y=0; y<3; y++)
      m(x,y) = OverlapB(d, occ[x], vir[y], alphainv, betainv, false);

  if (doparity) {
    cout << "not supported, SORRY!!"<<endl;
    exit(0);
  }
  return m.determinant();
}

double MoDeterminant::OverlapBBBB(Determinant& d, int i, int j, int k, int l,
				  int a, int b, int c, int e,
				  Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
				  bool doparity) {

  vector<int> occ(4,0); occ[0] = i; occ[1] = j; occ[2] = k; occ[3] = l;
  vector<int> vir(4,0); vir[0] = a; vir[1] = b; vir[2] = c; vir[3] = e;

  Matrix4f m ;

  for (int x=0; x<4; x++)
    for (int y=0; y<4; y++)
      m(x,y) = OverlapB(d, occ[x], vir[y], alphainv, betainv, false);

  if (doparity) {
    cout << "not supported, SORRY!!"<<endl;
    exit(0);
  }
  return m.determinant();
}


double MoDeterminant::OverlapBB(Determinant& d, int i, int j, int a, int b,
				Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
				bool doparity) {

  double p = 1.0;

  double factor1 = 1.0, factor2 = 1.0, factor3 = 1.0, factor4 = 1.0;

  factor1 = OverlapB(d, i, a, alphainv, betainv, false);
  factor2 = OverlapB(d, j, b, alphainv, betainv, false);
  factor3 = OverlapB(d, i, b, alphainv, betainv, false);
  factor4 = OverlapB(d, j, a, alphainv, betainv, false);
  if (doparity) {
    d.parityB(a, i, p); d.setoccB(a, true);  d.setoccB(i, false);
    d.parityB(b, j, p); d.setoccB(a, false); d.setoccB(i, true);
  }
  return p*(factor1*factor2 - factor3*factor4);
}

double MoDeterminant::OverlapAB(Determinant& d, int i, int j, int a, int b,
				Eigen::MatrixXd& alphainv, Eigen::MatrixXd &betainv,
				bool doparity) {

  double p = 1.0;

  double factor1 = 1.0, factor2 = 1.0, factor3 = 1.0, factor4 = 1.0;

  factor1 = OverlapA(d, i, a, alphainv, betainv, doparity);
  factor2 = OverlapB(d, j, b, alphainv, betainv, doparity);
  return p*(factor1*factor2);
}


void MoDeterminant::getDetMatrix(Determinant& d, MatrixXd& DetAlpha, MatrixXd &DetBeta) {
}

double MoDeterminant::Overlap(vector<int>& alpha, vector<int>& beta) {
  MatrixXd DetAlpha = MatrixXd::Zero(nalpha, nalpha);
  MatrixXd DetBeta  = MatrixXd::Zero(nbeta,  nbeta);

  //<psi1 | psi2> = det(phi1^dag phi2)
  //in out case psi1 is a simple occupation number determinant
  for (int i=0; i<alpha.size(); i++)
    for (int j=0; j<DetAlpha.cols(); j++)
      DetAlpha(i,j) = AlphaOrbitals(alpha[i], j);


  for (int i=0; i<beta.size(); i++)
    for (int j=0; j<DetBeta.cols(); j++)
      DetBeta(i,j) = BetaOrbitals(beta[i], j);

  double parity = 1.0;
  return parity*DetAlpha.determinant()*DetBeta.determinant();
}

void getunoccupiedOrbs(vector<int>& alpha, vector<int>& alphaOpen, int& norbs) {
  int i=0, j=0, index=0;
  while (i<alpha.size() && j < norbs) {
    if (alpha[i] < j) i++;
    else if (j<alpha[i]) {
      alphaOpen[index] = j;
      index++; j++;
    }
    else {i++;j++;}
  }
  while(j<norbs) {
    alphaOpen[index] = j;
    index++; j++;
  }
}

void MoDeterminant::HamAndOvlp(vector<int>& alpha, vector<int>& beta,
				 double& ovlp, double& ham,
				 oneInt& I1, twoInt& I2, double& coreE) {

  int norbs = MoDeterminant::norbs;
  vector<int> alphaOpen(norbs-alpha.size(),0), betaOpen(norbs-beta.size(),0);
  std::sort(alpha.begin(), alpha.end());
  std::sort(beta.begin() , beta.end() );

  getunoccupiedOrbs(alpha, alphaOpen, norbs);
  getunoccupiedOrbs(beta,  betaOpen,  norbs);


  //noexcitation
  {
    double E0 = coreE;
    for (int i=0; i<alpha.size(); i++) {
      int I = 2*alpha[i];
      E0 += I1(I, I);

      for (int j=i+1; j<alpha.size(); j++) {
	int J = 2*alpha[j];
	E0 += I2(I, I, J, J) - I2(I, J, I, J);
      }

      for (int j=0; j<beta.size(); j++) {
	int J = 2*beta[j]+1;
	E0 += I2(I, I, J, J);
      }
    }

    for (int i=0; i<beta.size(); i++) {
      int I = 2*beta[i]+1;
      E0 += I1(I, I);

      for (int j=i+1; j<beta.size(); j++) {
	int J = 2*beta[j]+1;
	E0 += I2(I, I, J, J) - I2(I, J, I, J);
      }
    }
    ovlp = Overlap(alpha, beta);
    ham  = ovlp*E0;
  }

  //Single excitation alpha
  {
    vector<int> alphaCopy = alpha;
    for (int i=0; i<alpha.size(); i++)
      for (int a=0; a<alphaOpen.size(); a++) {
	int I = 2*alpha[i], A = 2*alphaOpen[a];

	alphaCopy[i] = alphaOpen[a];
	double tia = I1(I, A);

	for (int j=0; j<alpha.size(); j++) {
	  int J = 2*alpha[j];
	  tia += I2(A, I, J, J) - I2(A, J, J, I);
	}
	for (int j=0; j<beta.size(); j++) {
	  int J = 2*beta[j]+1;
	  tia += I2(A, I, J, J);
	}

	if (abs(tia) > 1.e-8)
	  ham += tia*Overlap(alphaCopy, beta);
	alphaCopy[i] = alpha[i];
      }
  }

  //Single excitation beta
  {
    vector<int> betaCopy = beta;
    for (int i=0; i<beta.size(); i++)
      for (int a=0; a<betaOpen.size(); a++) {
	int I = 2*beta[i]+1, A = 2*betaOpen[a]+1;

	betaCopy[i] = betaOpen[a];
	double tia = I1(I, A);

	for (int j=0; j<beta.size(); j++) {
	  int J = 2*beta[j]+1;
	  tia += I2(A, I, J, J) - I2(A, J, J, I);
	}
	for (int j=0; j<alpha.size(); j++) {
	  int J = 2*alpha[j];
	  tia += I2(A, I, J, J);
	}

	if (abs(tia) > 1.e-8)
	  ham += tia*Overlap(alpha, betaCopy);

	betaCopy[i] = beta[i];
      }
  }



  //alpha-alpha
  {
    vector<int> alphaCopy = alpha;
    for (int i=0; i<alpha.size(); i++)
    for (int a=0; a<alphaOpen.size(); a++) {
      int I = 2*alpha[i], A = 2*alphaOpen[a];
      alphaCopy[i] = alphaOpen[a];

      for (int j=i+1; j<alpha.size(); j++)
      for (int b=a+1; b<alphaOpen.size(); b++) {
	int J = 2*alpha[j], B = 2*alphaOpen[b];

	double tiajb = I2(A, I, B, J) - I2(A, J, B, I);
	alphaCopy[j] = alphaOpen[b];

	if (abs(tiajb) > 1.e-8)
	  ham += tiajb*Overlap(alphaCopy, beta);
	alphaCopy[j] = alpha[j];
      }
      alphaCopy[i] = alpha[i];
    }
  }

  //alpha-beta
  {
    vector<int> alphaCopy = alpha;
    vector<int> betaCopy = beta;
    for (int i=0; i<alpha.size(); i++)
    for (int a=0; a<alphaOpen.size(); a++) {
      int I = 2*alpha[i], A = 2*alphaOpen[a];
      alphaCopy[i] = alphaOpen[a];
      for (int j=0; j<beta.size(); j++)
      for (int b=0; b<betaOpen.size(); b++) {
	int J = 2*beta[j]+1, B = 2*betaOpen[b]+1;

	double tiajb = I2(A, I, B, J);
	betaCopy[j] = betaOpen[b];

	if (abs(tiajb) > 1.e-8)
	  ham += tiajb*Overlap(alphaCopy, betaCopy);
	betaCopy[j] = beta[j];
      }
      alphaCopy[i] = alpha[i];
    }
  }


  //beta-beta
  {
    vector<int> betaCopy = beta;
    for (int i=0; i<beta.size(); i++)
    for (int a=0; a<betaOpen.size(); a++) {
      int I = 2*beta[i]+1, A = 2*betaOpen[a]+1;
      betaCopy[i] = betaOpen[a];

      for (int j=i+1; j<beta.size(); j++)
      for (int b=a+1; b<betaOpen.size(); b++) {
	int J = 2*beta[j]+1, B = 2*betaOpen[b]+1;

	double tiajb = I2(A, I, B, J) - I2(A, J, B, I);
	betaCopy[j] = betaOpen[b];

	if (abs(tiajb) > 1.e-8)
	  ham += tiajb*Overlap(alpha, betaCopy);
	betaCopy[j] = beta[j];
      }

      betaCopy[i] = beta[i];
    }
  }


}


//<d|*this>
double MoDeterminant::Overlap(MoDeterminant& d) {

  MatrixXd DetAlpha = d.AlphaOrbitals.transpose()*AlphaOrbitals;
  MatrixXd DetBeta  = d.BetaOrbitals.transpose() *BetaOrbitals;

  return DetAlpha.determinant()*DetBeta.determinant();

}
