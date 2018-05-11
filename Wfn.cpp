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


void CPSSlater::exciteWalker(Walker& wcopy, int excite1, int excite2, int norbs) {
  
  int I1 = excite1/(2*norbs), A1= excite1%(2*norbs);


  if (I1%2 == 0) wcopy.updateA(I1/2, A1/2, *this);
  else           wcopy.updateB(I1/2, A1/2, *this);

  if (excite2 != 0) {
    int I2 = excite2/(2*norbs), A2= excite2%(2*norbs);
    if (I2%2 == 0) wcopy.updateA(I2/2, A2/2, *this);
    else               wcopy.updateB(I2/2, A2/2, *this);
  }
  
}

void CPSSlater::PTcontribution2ndOrder(Walker& walk, double& E0,
				       oneInt& I1, twoInt& I2, 
				       twoIntHeatBathSHM& I2hb, double& coreE,
				       double& Aterm, double& Bterm, double& C,
				       vector<double>& ovlpRatio, vector<size_t>& excitation1, 
				       vector<size_t>& excitation2, bool doGradient) {

  auto random = std::bind(std::uniform_real_distribution<double>(0,1),
			  std::ref(generator));

  double TINY = schd.screen;
  double THRESH = schd.epsilon;


  double ovlp=0, ham=0;
  VectorXd grad;
  vector<Walker> firstRoundDets; vector<double> firstRoundCi; bool fillWalker = true;
  size_t nterms=schd.integralSampleSize;
  double coreEtmp = coreE - E0;


  firstRoundDets.push_back(walk); firstRoundCi.push_back(walk.d.Energy(I1, I2, coreEtmp));

  //Here we first calculate the loca energy of walker and then use that to select 
  //important elements of V
  double inputWalkerHam = 0;
  {
    vector<double> HijElements;
    HamAndOvlpGradient(walk, ovlp, inputWalkerHam, grad, I1, I2, I2hb, coreEtmp,
		       ovlpRatio, excitation1, excitation2, HijElements, doGradient);


    std::vector<size_t> idx(HijElements.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&HijElements](size_t i1, size_t i2)
	      {return abs(HijElements[i1])<abs(HijElements[i2]);});
    
    //pick the first nterms-1 most important integrals
    for (int i=0; i<min(nterms-1, HijElements.size()); i++) {
      int I = idx[HijElements.size()-i-1] ; //the largest Hij matrix elements

      firstRoundDets.push_back(walk);
      exciteWalker(*firstRoundDets.rbegin(), excitation1[I], excitation2[I], Determinant::norbs); 
      firstRoundCi.push_back(HijElements[I]*ovlpRatio[I]);

      //zero out the integral 
      HijElements[I] = 0;
    }

    //include the last term stochastically
    if (HijElements.size() > nterms-1) {

      double cumHij = 0;
      vector<double> cumHijList(HijElements.size(), 0);
      for (int i=0; i<HijElements.size(); i++) {
	cumHij += abs(HijElements[i]);
	cumHijList[i] = cumHij;
      }
      int T  = std::lower_bound(cumHijList.begin(), cumHijList.end(),
				random()*cumHij) - cumHijList.begin();
      firstRoundDets.push_back(walk);
      exciteWalker(*firstRoundDets.rbegin(), excitation1[T], excitation2[T], Determinant::norbs); 
      firstRoundCi.push_back(cumHij*ovlpRatio[T]*abs(HijElements[T])/HijElements[T]);
    }
  }


  //HamAndOvlpGradientStochastic(walk, ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
  //nterms, 
  //firstRoundDets, firstRoundCi, fillWalker);

  double Eidet = walk.d.Energy(I1, I2, coreE);
  C = 1.0/(Eidet-E0);
  Aterm = 0.0;

  for (int dindex =0; dindex<firstRoundDets.size(); dindex++) {
    double ovlp=0, ham=0;
    VectorXd grad;
    double Eidet = firstRoundDets[dindex].d.Energy(I1, I2, coreE);

    
    if (dindex == 0)  {
      ham = inputWalkerHam;
      //vector<double> HijElements;
      //HamAndOvlpGradient(firstRoundDets[dindex], ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
      //ovlpRatio, excitation1, excitation2, HijElements, doGradient);
      Bterm = ham/(Eidet-E0);
    }
    else {
      vector<double> HijElements;
      vector<double> ovlpRatio; vector<size_t> excitation1, excitation2;
      HamAndOvlpGradient(firstRoundDets[dindex], ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
			 ovlpRatio, excitation1, excitation2, HijElements, doGradient, false);
    }

    Aterm -= (ham)*firstRoundCi[dindex]/(Eidet - E0);
    //cout << firstRoundCi[dindex]<<"  "<<firstRoundDets[dindex].d<<"  "<<ham<<"  "<<Aterm<<endl;
  }

}


void CPSSlater::PTcontribution3rdOrder(Walker& walk, double& E0,
				       oneInt& I1, twoInt& I2, 
				       twoIntHeatBathSHM& I2hb, double& coreE,
				       double& Aterm2, double& Bterm, double& C, double& Aterm3,
				       vector<double>& ovlpRatio, vector<size_t>& excitation1, 
				       vector<size_t>& excitation2, bool doGradient) {
  

  double TINY = schd.screen;
  double THRESH = schd.epsilon;


  double ovlp=0, ham=0;
  VectorXd grad;
  vector<Walker> firstRoundDets; vector<double> firstRoundCi; bool fillWalker = true;
  int nterms=schd.integralSampleSize;

  double coreEtmp = coreE - E0;
  HamAndOvlpGradientStochastic(walk, ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
			       nterms, 
			       firstRoundDets, firstRoundCi, fillWalker);

  double Eidet = walk.d.Energy(I1, I2, coreE);
  C = 1.0/(Eidet-E0);
  Aterm2 = 0.0; Aterm3 = 0.0;
  double sgn = ovlp/abs(ovlp);

  for (int dindex1 =0; dindex1<firstRoundDets.size(); dindex1++) {
    double Eidet = firstRoundDets[dindex1].d.Energy(I1, I2, coreE);
    firstRoundCi[dindex1] /= (Eidet- E0);

    vector<Walker> secondRoundDets; vector<double> secondRoundCi; bool fillWalker = true;
    HamAndOvlpGradientStochastic(firstRoundDets[dindex1], ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
				 nterms, secondRoundDets, secondRoundCi, fillWalker);
    
				 
    for (int dindex2 =0; dindex2<secondRoundDets.size(); dindex2++) {
      double ovlp=0, ham=0;
      double Eidet = secondRoundDets[dindex2].d.Energy(I1, I2, coreE);
      secondRoundCi[dindex2] *= firstRoundCi[dindex1] / (Eidet- E0);

    
      if (dindex2 == 0 && dindex1 == 0)  {
	vector<double> HijElements;
	HamAndOvlpGradient(secondRoundDets[dindex2], ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
			   ovlpRatio, excitation1, excitation2, HijElements, doGradient);
	Bterm = ham/(Eidet-E0);
      }
      else {
	vector<double> HijElements;
	vector<double> ovlpRatio; vector<size_t> excitation1, excitation2;
	HamAndOvlpGradient(secondRoundDets[dindex2], ovlp, ham, grad, I1, I2, I2hb, coreEtmp,
			   ovlpRatio, excitation1, excitation2, HijElements, doGradient, false);
      }
    
      if (dindex2 == 0) 
	Aterm2 -= (ham)*firstRoundCi[dindex1];
      
      Aterm3 -= (ham)*secondRoundCi[dindex2];
    }
  }
}

//only appends to ham and returnWalkder, coeffWalker, so make sure they are empty at the
//begining if you don't want to just append things
//it gives the following output, where i can be nsingles+ndoubles
// <psi|H|D_j> = \sum_i (H_ij/pi) *(<psi|D_i>/<psi|D_j>) <psi|D_j>
void CPSSlater::HamAndOvlpGradientStochastic(Walker& walk,
					     double& ovlp, double& ham, VectorXd& grad,
					     oneInt& I1, twoInt& I2, 
					     twoIntHeatBathSHM& I2hb, double& coreE,
					     int nterms,
 					     vector<Walker>& returnWalker, 
					     vector<double>& coeffWalker, 
					     bool fillWalker) {

  double TINY = schd.screen;
  double THRESH = schd.epsilon;

  MatrixXd alphainv = walk.alphainv, betainv = walk.betainv;
  double alphaDet = walk.alphaDet, betaDet = walk.betaDet;
  double detOverlap = alphaDet*betaDet;
  ovlp = detOverlap;
  for (int i=0; i<cpsArray.size(); i++)
    ovlp *= cpsArray[i].Overlap(walk.d);
  
  {
    double E0 = walk.d.Energy(I1, I2, coreE);
    ham  += E0;

    if (fillWalker) {
      returnWalker.push_back(walk);
      coeffWalker .push_back(E0);
    }
  }


  vector<int> Isingle, Asingle;
  vector<int> Idouble, Adouble,
    Jdouble, Bdouble;

  vector<double> psingle,
    pdouble;
  

  sampleSingleDoubleExcitation(walk.d, I1, I2, I2hb, 
			       nterms,
			       Isingle, Asingle,
			       Idouble, Adouble,
			       Jdouble, Bdouble,
			       psingle, 
			       pdouble);


  //cout << Isingle.size()<<"  "<<Idouble.size()<<"  ";

  //Calculate the contribution to the matrix through the single 
  double HijSingle = 0;
  int nsingles = Isingle.size();
  for (int ns=0; ns<nsingles; ns++)
  {  
    Determinant dcopy = walk.d;
    dcopy.setocc(Isingle[ns], false); dcopy.setocc(Asingle[ns], true);
    int I = Isingle[ns]/2, A = Asingle[ns]/2;

    HijSingle = Isingle[ns]%2 == 0 ? 
      walk.d.Hij_1ExciteA(A, I, I1, I2)
      *det.OverlapA(walk.d, I, A, alphainv, betainv) :
      
      walk.d.Hij_1ExciteB(A, I, I1, I2) 
      *det.OverlapB(walk.d, I, A, alphainv, betainv) ;
    
    for (int n=0; n<cpsArray.size(); n++) 
      if (std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), I) ||
	  std::binary_search(cpsArray[n].asites.begin(), cpsArray[n].asites.end(), A) )
	HijSingle *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(walk.d);
    
    ham += HijSingle/psingle[ns]/nterms;

    if (fillWalker) {
      Walker wcopy = walk;
      if (Isingle[ns]%2 == 0) wcopy.updateA(Isingle[ns]/2, Asingle[ns]/2, *this);
      else                    wcopy.updateB(Isingle[ns]/2, Asingle[ns]/2, *this);
      
      returnWalker.push_back(wcopy);
      coeffWalker .push_back(HijSingle/psingle[ns]/nterms);
    }
  }



  int ndoubles = Idouble.size();
  //Calculate the contribution to the matrix through the double 
  for (int nd=0; nd<ndoubles; nd++) {
    double Hijdouble = (I2(Adouble[nd], Idouble[nd], Bdouble[nd], Jdouble[nd]) 
			- I2(Adouble[nd], Jdouble[nd], Bdouble[nd], Idouble[nd]));

    Determinant dcopy = walk.d;
    dcopy.setocc(Idouble[nd], false); dcopy.setocc(Adouble[nd], true);
    dcopy.setocc(Jdouble[nd], false); dcopy.setocc(Bdouble[nd], true);
    
    int I = Idouble[nd]/2, J = Jdouble[nd]/2, A = Adouble[nd]/2, B = Bdouble[nd]/2;

    if      (Idouble[nd]%2==Jdouble[nd]%2 && Idouble[nd]%2 == 0) 
      Hijdouble *= det.OverlapAA(walk.d, I, J, A, B,  alphainv, betainv, false);

    else if (Idouble[nd]%2==Jdouble[nd]%2 && Idouble[nd]%2 == 1) 
      Hijdouble *= det.OverlapBB(walk.d, I, J, A, B,  alphainv, betainv, false);

    else if (Idouble[nd]%2!=Jdouble[nd]%2 && Idouble[nd]%2 == 0)
      Hijdouble *= det.OverlapAB(walk.d, I, J, A, B,  alphainv, betainv, false);

    else
      Hijdouble *= det.OverlapAB(walk.d, J, I, B, A,  alphainv, betainv, false);

    for (int n=0; n<cpsArray.size(); n++) 
      for (int j = 0; j<cpsArray[n].asites.size(); j++) {
	if (cpsArray[n].asites[j] == I || 
	    cpsArray[n].asites[j] == J || 
	    cpsArray[n].asites[j] == A ||
	    cpsArray[n].asites[j] == B) {
	  Hijdouble *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(walk.d);
	  //localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);
	  break;
	}
      }

    ham += Hijdouble/pdouble[nd]/nterms;

    if (fillWalker) {
      Walker wcopy = walk;
      if (Idouble[nd]%2 == 0) wcopy.updateA(Idouble[nd]/2, Adouble[nd]/2, *this);
      else                wcopy.updateB(Idouble[nd]/2, Adouble[nd]/2, *this);
      if (Jdouble[nd]%2 == 0) wcopy.updateA(Jdouble[nd]/2, Bdouble[nd]/2, *this);
      else                wcopy.updateB(Jdouble[nd]/2, Bdouble[nd]/2, *this);
      
      returnWalker.push_back(wcopy);
      coeffWalker .push_back(Hijdouble/pdouble[nd]/nterms);
    }
  }
}


//<psi_t| (H-E0) |D>
void CPSSlater::HamAndOvlpGradient(Walker& walk,
				   double& ovlp, double& ham, VectorXd& grad,
				   oneInt& I1, twoInt& I2, 
				   twoIntHeatBathSHM& I2hb, double& coreE,
				   vector<double>& ovlpRatio, vector<size_t>& excitation1, 
				   vector<size_t>& excitation2, vector<double>& HijElements, 
				   bool doGradient, bool fillExcitations) {
  ovlpRatio.clear(); excitation1.clear(); excitation2.clear();

  double TINY = schd.screen;
  double THRESH = schd.epsilon;
  MatrixXd alphainv = walk.alphainv, betainv = walk.betainv;
  double alphaDet = walk.alphaDet, betaDet = walk.betaDet;
  double detOverlap = alphaDet*betaDet;
  Determinant& d = walk.d;

  int norbs = Determinant::norbs;
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
    //if (commrank == 0) cout << ham<<endl;

    if (doGradient) {
      double factor = E0;    
      OverlapWithGradient(d, factor, grad);
    }
  }
  //cout << ham<<endl;

  //Single alpha-beta excitation 
  {
    for (int i=0; i<closed.size(); i++) {
      for (int a=0; a<open.size(); a++) {
	if (closed[i]%2 == open[a]%2 && I2hb.Singles(closed[i], open[a]) > TINY) {
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
	    //if (commrank == 0) cout << ham<<"  s "<<endl;

	    //cout << ham<<"  "<<localham<<endl;
	    //cout << ham<<endl;
	    
	    double ovlpdetcopy = localham/tia;
	    if (doGradient) {
	      double factor = tia * ovlpdetcopy ;
	      OverlapWithGradient(dcopy, factor, grad);
	    }

	    if (fillExcitations) {
	      ovlpRatio.push_back(ovlpdetcopy);
	      excitation1.push_back( closed[i]*2*norbs+open[a]);
	      excitation2.push_back(0);
	      HijElements.push_back(tia);
	    }
	    //cout << *excitation1.rbegin()<<"  "<<*excitation2.rbegin()<<"  "<<excitation1.size()<<"  "<<excitation2.size()<<endl;
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
	if (fabs(integrals[index]) < THRESH) break;

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
	    for (int j = 0; j<cpsArray[n].asites.size(); j++) {
	      if (cpsArray[n].asites[j] == I || 
		  cpsArray[n].asites[j] == J || 
		  cpsArray[n].asites[j] == A ||
		  cpsArray[n].asites[j] == B) {
		localham *= cpsArray[n].Overlap(dcopy)/cpsArray[n].Overlap(d);
		break;
	      }
	    }

	  ham += localham;
	  //cout << ham<<"  "<<localham<<endl;
	  
	  double ovlpdetcopy = localham/tiajb;
	  if (doGradient) {
	    double factor = tiajb * ovlpdetcopy;
	    OverlapWithGradient(dcopy, factor, grad);
	  }

	  if (fillExcitations) {
	    ovlpRatio.push_back(ovlpdetcopy);
	    excitation1.push_back( closed[i]*2*norbs+a);
	    excitation2.push_back( closed[j]*2*norbs+b);
	    HijElements.push_back(tiajb);
	  }
	  //cout << *excitation1.rbegin()<<"  "<<*excitation2.rbegin()<<"  "<<excitation1.size()<<"  "<<excitation2.size()<<endl;

	}
      }
    }

  }
}


