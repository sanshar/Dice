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

#ifndef Lanczos_HEADER_H
#define Lanczos_HEADER_H
#include <vector>
#include <set>
#include "Determinants.h"
#include "workingArray.h"
#include "excitationOperators.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <utility>

using namespace std;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;


template<typename Wfn>
class Lanczos
{
 private:
  friend class boost::serialization::access;
  template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
      ar  & wave
	& alpha;
    }

 public:
  double alpha;
  Wfn wave;
  workingArray morework;
  
  Lanczos()
  {
    wave.readWave();
    alpha = 0.;
  }

  Lanczos(Wfn &w1, double alpha0) : alpha(alpha0)
  {
    wave = w1;
  }

  typename Wfn::ReferenceType& getRef() { return wave.getRef(); }
  typename Wfn::CorrType& getCorr() { return wave.getCorr(); }

  template<typename Walker>
  void initWalker(Walker& walk) {
    this->wave.initWalker(walk);
  }  

  template<typename Walker>
  double optimizeWave(Walker& walk, double alphaInit = 0.1) {
    Eigen::VectorXd stddev = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd rk = Eigen::VectorXd::Zero(4);
    //double rk = 0;
    Eigen::VectorXd lanczosCoeffs = Eigen::VectorXd::Zero(4);
    alpha = alphaInit;
    if (schd.deterministic) getLanczosCoeffsDeterministic(wave, walk, alpha, lanczosCoeffs);
    else getLanczosCoeffsContinuousTime(wave, walk, alpha, lanczosCoeffs, stddev, rk, schd.stochasticIter, 1.e-5);

    double a = lanczosCoeffs[2]/lanczosCoeffs[3];
    double b = lanczosCoeffs[1]/lanczosCoeffs[3];
    double c = lanczosCoeffs[0]/lanczosCoeffs[3];
    double delta = pow(a, 2) + 4 * pow(b, 3) - 6 * a * b * c - 3 * pow(b * c, 2) + 4 * a * pow(c, 3);
    double numP = a - b * c + pow(delta, 0.5);
    double numM = a - b * c - pow(delta, 0.5);
    double denom = 2 * pow(b, 2) - 2 * a * c;
    double alphaP = numP/denom;
    double alphaM = numM/denom;
    double eP = (a * pow(alphaP, 2) + 2 * b * alphaP + c) / (b * pow(alphaP, 2) + 2 * c * alphaP + 1);
    double eM = (a * pow(alphaM, 2) + 2 * b * alphaM + c) / (b * pow(alphaM, 2) + 2 * c * alphaM + 1);
    if (commrank == 0) {
      cout << "coeffs\n" << lanczosCoeffs << endl; 
      //cout << "a  " << a << "  b  " << b << "  b  " << "   c  " << c << endl;
      cout << "alpha(+/-)   " << alphaP << "   " << alphaM << endl;
      cout << "energy(+/-)   " << eP << "   " << eM << endl;
    }
    if (eP < eM) alpha = alphaP;
    else alpha = alphaM;
    return alpha;
  }
  
  template<typename Walker>
  void initWalker(Walker& walk, Determinant& d) {
    this->wave.initWalker(walk, d);
  }  

  void getVariables(VectorXd& vars) {
    if (vars.rows() != getNumVariables())
      {
	vars = VectorXd::Zero(getNumVariables());
      }
      vars[0] = alpha;
  }

  void printVariables() {
    cout << "alpha  " << alpha << endl;
  }

  void updateVariables(VectorXd& vars) {
    alpha = vars[0];
  }

  long getNumVariables() {
    return 1;
  }
  
  template<typename Walker>
  double Overlap(Walker& walk) {
    int norbs = Determinant::norbs;
    double totalovlp = 0.0;

    double ovlp = wave.Overlap(walk);
    double ham = walk.d.Energy(I1, I2, coreE); 

    morework.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        morework, false);  
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        morework, false);  
  
    //loop over all the screened excitations
    for (int i=0; i<morework.nExcitations; i++) {
      int ex1 = morework.excitation1[i], ex2 = morework.excitation2[i];
      double tia = morework.HijElement[i];
    
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

      double ovlpRatio = wave.getOverlapFactor(I, J, A, B, walk, false);

      ham += tia * ovlpRatio;

      morework.ovlpRatio[i] = ovlpRatio;
    }

    totalovlp = ovlp*(1 + alpha * ham);
    return totalovlp;
  }

  template<typename Walker>
  void HamAndOvlp(const Walker &walk,
                  double &ovlp, double& ham,
                  workingArray& work, bool fillExcitations = true)
  {
    int norbs = Determinant::norbs;

    ham = walk.d.Energy(I1, I2, coreE); 
    //ovlp = Overlap(walk);

    morework.setCounterToZero();
    double ovlp0, ham0;
    wave.HamAndOvlp(walk, ovlp0, ham0, morework);
    ovlp = ovlp0*(1 + alpha*ham0);

    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);  

    //loop over all the screened excitations
    for (int i=0; i<work.nExcitations; i++) {
      double tia = work.HijElement[i];
      auto walkCopy = walk;
      walkCopy.updateWalker(wave.getRef(), wave.getCorr(),
                            work.excitation1[i], work.excitation2[i], false);

      //double ovlp0 = Overlap(walkCopy);

      morework.setCounterToZero();
      wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework);
      ovlp0 = ovlp0*(1 + alpha*ham0);

      ham += tia * ovlp0/ovlp;
      work.ovlpRatio[i] = ovlp0/ovlp;
    }
  }

  string getfileName() const {
    return "lanczos"+wave.getfileName();
  }

  void writeWave()
  {
    if (commrank == 0)
      {
	char file[5000];
        sprintf(file, (getfileName()+".bkp").c_str() );
	//sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
	//sprintf(file, "lanczoscpswave.bkp");
	std::ofstream outfs(file, std::ios::binary);
	boost::archive::binary_oarchive save(outfs);
	save << *this;
	outfs.close();
      }
  }

  void readWave()
  {
    if (commrank == 0)
      {
	char file[5000];
        sprintf(file, (getfileName()+".bkp").c_str() );
	//sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
	//sprintf(file, "lanczoscpswave.bkp");
	std::ifstream infs(file, std::ios::binary);
	boost::archive::binary_iarchive load(infs);
	load >> *this;
	infs.close();
      }
#ifndef SERIAL
    boost::mpi::communicator world;
    boost::mpi::broadcast(world, *this, 0);
#endif
  }
};
#endif
