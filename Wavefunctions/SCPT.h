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

#ifndef SCPT_HEADER_H
#define SCPT_HEADER_H
#include <vector>
#include <set>
#include "Determinants.h"
#include "workingArray.h"
#include "excitationOperators.h"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <Eigen/Eigenvalues>
#include <utility>
#include <iomanip>

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;


template<typename Wfn>
class SCPT
{
 private:
  friend class boost::serialization::access;
  template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
      ar  & wave
	& coeffs
    & moEne;
    }

 public:
  VectorXd coeffs;
  VectorXd moEne;
  Wfn wave; //reference wavefunction
  workingArray morework;
  

  SCPT()
  {
    wave.readWave();

    // Resize coeffs
    int numVirt = Determinant::norbs - schd.nciAct;
    int numCoeffs = 1 + 2*numVirt + (2*numVirt * (2*numVirt - 1) / 2);
    coeffs = VectorXd::Zero(numCoeffs);
    moEne = VectorXd::Zero(Determinant::norbs);

    //coeffs order: phi0, singly excited (spin orb index), doubly excited (spin orb pair index)

    if (commrank == 0) {
      auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                              std::ref(generator));

      coeffs(0) = -0.5;
      for (int i=1; i < numCoeffs; i++) {
        coeffs(i) = 0.2*random() - 0.1;
      }
    }

#ifndef SERIAL
  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    
    char file[5000];
    sprintf(file, "ciCoeffs.txt");
    ifstream ofile(file);
    if (ofile) {
      for (int i = 0; i < coeffs.size(); i++) {
        ofile >> coeffs(i);
      }
    }
    
    char filem[5000];
    sprintf(filem, "moEne.txt");
    ifstream ofilem(filem);
    if (ofilem) {
      for (int i = 0; i < Determinant::norbs; i++) {
        ofilem >> moEne(i);
      }
    }
    else {
      if (commrank == 0) cout << "moEne.txt not found!\n";
      exit(0);
    }
  }

  typename Wfn::ReferenceType& getRef() { return wave.getRef(); }
  typename Wfn::CorrType& getCorr() { return wave.getCorr(); }

  template<typename Walker>
  void initWalker(Walker& walk) {
    this->wave.initWalker(walk);
  }
  
  template<typename Walker>
  void initWalker(Walker& walk, Determinant& d) {
    this->wave.initWalker(walk, d);
  }
  
  //void initWalker(Walker& walk) {
  //  this->wave.initWalker(walk);
  //}

  void getVariables(VectorXd& vars) {
    vars = coeffs;
  }

  void printVariables() {
    cout << "ci coeffs\n" << coeffs << endl;
  }

  void updateVariables(VectorXd& vars) {
    coeffs = vars;
  }

  long getNumVariables() {
    return coeffs.size();
  }

  template<typename Walker>
  int coeffsIndex(Walker& walk) {
    int norbs = Determinant::norbs;
    if (walk.excitedOrbs.size() == 2) {
      int i = *walk.excitedOrbs.begin() - 2*schd.nciAct;
      int j = *(std::next(walk.excitedOrbs.begin())) - 2*schd.nciAct;
      int I = max(i, j) - 1, J = min(i,j);
      int numVirt = norbs - schd.nciAct;
      return 1 + 2*numVirt + I*(I+1)/2 + J;
    }
    else if (walk.excitedOrbs.size() == 1) {
      return *walk.excitedOrbs.begin() - 2*schd.nciAct + 1;
    }
    else if (walk.excitedOrbs.size() == 0) return 0;
    else return -1;
  }
  
  template<typename Walker>
  double getOverlapFactor(int i, int a, const Walker& walk, bool doparity) const  
  {
    return 1.;
  }//not used
  
  template<typename Walker>
  double getOverlapFactor(int I, int J, int A, int B, const Walker& walk, bool doparity) const 
  {
    return 1.;
  }//not used
  
  //ham is a sample of the diagonal element of the Dyall ham
  template<typename Walker>
  void HamAndOvlp(Walker &walk,
                  double &ovlp, double &locEne, double &ham, double &norm, int coeffsIndex, 
                  workingArray& work, bool fillExcitations=true) 
  {
    int norbs = Determinant::norbs;
    double ciCoeff = coeffs(coeffsIndex);
    morework.setCounterToZero();
    double ovlp0, ham0;
    if (coeffsIndex == 0) {
      wave.HamAndOvlp(walk, ovlp0, ham0, morework, true);
      ovlp = ciCoeff * ovlp0;
    }
    else {
      wave.HamAndOvlp(walk, ovlp0, ham0, morework, false);
      ovlp = ciCoeff * ovlp0;
    }
    if (ovlp == 0.) return; //maybe not necessary
    if (abs(ciCoeff) < 1.e-5) norm = 0;
    else norm = 1 / ciCoeff / ciCoeff;
    locEne = walk.d.Energy(I1, I2, coreE);
    double dEne = locEne;
    Determinant dAct = walk.d;
    //cout << "walker\n" << walk << endl;
    //cout << "ovlp  " << ovlp << endl;
    if (walk.excitedOrbs.size() == 0) {
      ham = walk.d.Energy(I1, I2, coreE) / ciCoeff / ciCoeff;
      //cout << "ene  " << ham << endl;
    }
    else {
      dAct.setocc(*walk.excitedOrbs.begin(), false);
      double ene1 = moEne((*walk.excitedOrbs.begin())/2);
      if (walk.excitedOrbs.size() == 1) {
        ham = (dAct.Energy(I1, I2, coreE) + ene1) / ciCoeff / ciCoeff;
        //cout << "ene  " << ham << endl;
      }
      else {
        dAct.setocc(*(std::next(walk.excitedOrbs.begin())), false);
        double ene2 = moEne((*(std::next(walk.excitedOrbs.begin())))/2);
        ham = (dAct.Energy(I1, I2, coreE) + ene1 + ene2) / ciCoeff / ciCoeff;
        //cout << "ene  " << ham << endl;
      }
    }

    work.setCounterToZero();
    generateAllScreenedSingleExcitationsDyall(walk.d, dAct, schd.epsilon, schd.screen,
                                        work, false);
    generateAllScreenedDoubleExcitationsDyall(walk.d, schd.epsilon, schd.screen,
                                        work, false);

    //loop over all the screened excitations
    //cout << endl << "m dets\n" << endl << endl;
    for (int i=0; i<work.nExcitations; i++) {
      double tia = work.HijElement[i];
      double tiaD = work.HijElement[i];
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      double isDyall = 0.;
      if (work.ovlpRatio[i] != 0.) {
        isDyall = 1.;
        if (ex2 == 0) tiaD = work.ovlpRatio[i];
        work.ovlpRatio[i] = 0.;
      }
      
      auto walkCopy = walk;
      double parity = 1.;
      Determinant dcopy = walkCopy.d;
      walkCopy.updateWalker(wave.getRef(), wave.getCorr(),
                            work.excitation1[i], work.excitation2[i], false);
      //cout << walkCopy << endl;
      if (walkCopy.excitedOrbs.size() > 2) continue;
      parity *= dcopy.parity(A/2, I/2, I%2);
      //if (ex2 == 0) {
      //  ham0 = dEne + walk.energyIntermediates[A%2][A/2] - walk.energyIntermediates[I%2][I/2] 
      //              - (I2.Direct(I/2, A/2) - I2.Exchange(I/2, A/2));
      //}
      //else {
      if (ex2 != 0) {
        dcopy.setocc(I, false);
        dcopy.setocc(A, true);
        parity *= dcopy.parity(B/2, J/2, J%2);
        //bool sameSpin = (I%2 == J%2);
        //ham0 = dEne + walk.energyIntermediates[A%2][A/2] - walk.energyIntermediates[I%2][I/2]
        //            + walk.energyIntermediates[B%2][B/2] - walk.energyIntermediates[J%2][J/2]
        //            + I2.Direct(A/2, B/2) - sameSpin * I2.Exchange(A/2, B/2)
        //            + I2.Direct(I/2, J/2) - sameSpin * I2.Exchange(I/2, J/2)
        //            - (I2.Direct(I/2, A/2) - I2.Exchange(I/2, A/2))
        //            - (I2.Direct(J/2, B/2) - I2.Exchange(J/2, B/2))
        //            - (I2.Direct(I/2, B/2) - sameSpin * I2.Exchange(I/2, B/2))
        //            - (I2.Direct(J/2, A/2) - sameSpin * I2.Exchange(J/2, A/2));
      }
      
      int coeffsCopyIndex = this->coeffsIndex(walkCopy);
      morework.setCounterToZero();
      //cout << "ovlp  " << ovlp0 << "  ham  " << ham0 << "  tia  " << tia << "  parity  " << parity << endl;
      if (coeffsCopyIndex == 0) {
        wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework, true);
        ham += isDyall * parity * tiaD * ovlp0 / ciCoeff / ovlp;
        locEne += parity * tia * ovlp0 * coeffs(coeffsCopyIndex) / ovlp;
        work.ovlpRatio[i] = ovlp0 * coeffs(coeffsCopyIndex) / ovlp;
      }
      else {
        wave.HamAndOvlp(walkCopy, ovlp0, ham0, morework, false);
        ham += isDyall * parity * tiaD * ham0 / ciCoeff / ovlp;
        locEne += parity * tia * ham0 * coeffs(coeffsCopyIndex) / ovlp;
        work.ovlpRatio[i] = ham0 * coeffs(coeffsCopyIndex) / ovlp;
      }
      //cout << endl;
    }
    //cout << endl << "n ham  " << ham << "  norm  " << norm << endl << endl;
  }
  
  template<typename Walker>
  double optimizeWaveCT(Walker& walk) {
    
    //add noise to avoid zero coeffs
    if (commrank == 0) {
      //cout << "starting sampling at " << setprecision(4) << getTime() - startofCalc << endl; 
      auto random = std::bind(std::uniform_real_distribution<double>(0., 1.e-6), std::ref(generator));
      for (int i=0; i < coeffs.size(); i++) {
        if (coeffs(i) == 0) coeffs(i) = random();
      }
    }

#ifndef SERIAL
  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    //sampling
    int norbs = Determinant::norbs;
    auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                            std::ref(generator));

    double ovlp = 0., normSample = 0., hamSample = 0., locEne = 0., waveEne = 0., correctionFactor = 0.;
    VectorXd ham = VectorXd::Zero(coeffs.size()), norm = VectorXd::Zero(coeffs.size());
    int coeffsIndex = this->coeffsIndex(walk);
    workingArray work;
    HamAndOvlp(walk, ovlp, locEne, hamSample, normSample, coeffsIndex, work);

    int iter = 0;
    double cumdeltaT = 0.;
    int printMod = schd.stochasticIter / 5;

    while (iter < schd.stochasticIter) {
      double cumovlpRatio = 0;
      for (int i = 0; i < work.nExcitations; i++) {
        cumovlpRatio += abs(work.ovlpRatio[i]);
        work.ovlpRatio[i] = cumovlpRatio;
      }
      double deltaT = 1.0 / (cumovlpRatio);
      double nextDetRandom = random() * cumovlpRatio;
      int nextDet = std::lower_bound(work.ovlpRatio.begin(), (work.ovlpRatio.begin() + work.nExcitations),
                                     nextDetRandom) - work.ovlpRatio.begin();
      cumdeltaT += deltaT;
      double ratio = deltaT / cumdeltaT;
      norm *= (1 - ratio);
      norm(coeffsIndex) += ratio * normSample;
      ham *= (1 - ratio);
      ham(coeffsIndex) += ratio * hamSample;
      waveEne *= (1 - ratio);
      waveEne += ratio * locEne;
      correctionFactor *= (1 - ratio);
      if (coeffsIndex == 0) {
        correctionFactor += ratio;
      }
      walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
      coeffsIndex = this->coeffsIndex(walk);
      HamAndOvlp(walk, ovlp, locEne, hamSample, normSample, coeffsIndex, work);
      iter++;
      if (commrank == 0 && iter % printMod == 1) cout << "iter  " << iter << "  t  " << setprecision(4) << getTime() - startofCalc << endl; 
    }
  
    norm *= cumdeltaT;
    ham *= cumdeltaT;
    waveEne *= cumdeltaT;
    correctionFactor *= cumdeltaT;

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, norm.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ham.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(cumdeltaT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(waveEne), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(correctionFactor), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    norm /= cumdeltaT;
    ham /= cumdeltaT;
    waveEne /= cumdeltaT;
    correctionFactor /= cumdeltaT;
    
    std::vector<int> largeNormIndices;
    int counter = 0;
    for (int i = 0; i < coeffs.size(); i++) {
      if (norm(i) > schd.overlapCutoff) {
        largeNormIndices.push_back(i);
      }
    }
    Map<VectorXi> largeNormSlice(&largeNormIndices[0], largeNormIndices.size());
    VectorXd largeNorms;
    igl::slice(norm, largeNormSlice, largeNorms);
    VectorXd largeHam;
    igl::slice(ham, largeNormSlice, largeHam);
    VectorXd ene = (largeHam.array() / largeNorms.array()).matrix();
    double ene2 = 0.;
    coeffs.setZero();
    coeffs(0) = 1.;
    //if (commrank == 0) {
    //  cout << "correctionFactor  " << correctionFactor << endl;
    //  cout << "i   largeNorms(i)    ene(0) - ene(i)\n";
    //}
    for (int i = 1; i < largeNorms.size(); i++) {
      //ene2 += largeNorms(i) / correctionFactor / (ene(0) - ene(i));
      ene2 += largeNorms(i) / largeNorms(0) / (ene(0) - ene(i));
      //if (commrank == 0) cout << i << "     " << largeNorms(i) << "    " << ene(0) - ene(i) << endl;
      coeffs(largeNormIndices[i]) = 1 / (ene(0) - ene(i));
    }
    
    if (commrank == 0) {
      cout << "ref energy   " << setprecision(12) << ene(0) << endl;
      cout << "nevpt2 energy  " << ene(0) + ene2 << endl;
      cout << "waveEne  " << waveEne << endl;
      if (schd.printVars) cout << endl << "ci coeffs\n" << coeffs << endl; 
    }
  }
  
  template<typename Walker>
  double optimizeWaveDeterministic(Walker& walk) {

    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    vector<Determinant> allDets;
    generateAllDeterminants(allDets, norbs, nalpha, nbeta);

    workingArray work;
    double overlapTot = 0., correctionFactor = 0.; 
    VectorXd ham = VectorXd::Zero(coeffs.size()), norm = VectorXd::Zero(coeffs.size());
    double waveEne = 0.;
    //w.printVariables();

    for (int i = commrank; i < allDets.size(); i += commsize) {
      wave.initWalker(walk, allDets[i]);
      if (schd.debug) {
        cout << "walker\n" << walk << endl;
      }
      if (walk.excitedOrbs.size() > 2) continue;
      int coeffsIndex = this->coeffsIndex(walk);
      double ovlp = 0., normSample = 0., hamSample = 0., locEne = 0.;
      HamAndOvlp(walk, ovlp, locEne, hamSample, normSample, coeffsIndex, work);
      //cout << "ham  " << ham[0] << "  " << ham[1] << "  " << ham[2] << endl;
      //cout << "ovlp  " << ovlp[0] << "  " << ovlp[1] << "  " << ovlp[2] << endl << endl;
      
      overlapTot += ovlp * ovlp;
      ham(coeffsIndex) += (ovlp * ovlp) * hamSample;
      norm(coeffsIndex) += (ovlp * ovlp) * normSample;
      waveEne += (ovlp * ovlp) * locEne;
      if (coeffsIndex == 0) correctionFactor += ovlp * ovlp;
    }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(overlapTot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(correctionFactor), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(waveEne), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ham.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, norm.data(), coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    VectorXd ene = VectorXd::Zero(coeffs.size());
    if (commrank == 0) {
      waveEne = waveEne / overlapTot;
      cout << "waveEne  " << waveEne << endl;
      //cout << "overlapTot  " << overlapTot << endl;
      ham = ham / overlapTot;
      //cout << "ham\n" << ham << endl;
      norm = norm / overlapTot;
      // cout << "norm\n" << norm << endl;
      ene = (ham.array() / norm.array()).matrix();
      correctionFactor = correctionFactor / overlapTot;
      double ene2 = 0.;
      for (int i = 1; i < coeffs.size(); i++) ene2 += norm(i) / correctionFactor / (ene(0) - ene(i));
      cout << "nevpt2 energy  " << ene(0) + ene2 << endl;
      //cout << "ene\n" << ene << endl;
      coeffs(0) = 1.;
      for (int i = 1; i < coeffs.size(); i++) coeffs(i) = 1 / (ene(0) - ene(i));
    }
#ifndef SERIAL
  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //MPI_Bcast(&(ene0), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }

  string getfileName() const {
    return "scci"+wave.getfileName();
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
