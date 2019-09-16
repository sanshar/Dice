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

#ifndef EOM_HEADER_H
#define EOM_HEADER_H
#include <vector>
#include <set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <utility>
#include <iomanip>
#include "CorrelatedWavefunction.h"

#ifndef SERIAL
#include "mpi.h"
#endif

using namespace std;

class oneInt;
class twoInt;
class twoIntHeatBathSHM;


template<typename Corr, typename Reference>
class EOM
{
 private:
  friend class boost::serialization::access;
  template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
      ar  & wave
	  & coeffs;
    }

 public:
  using CorrType = Corr;
  using ReferenceType = Reference;
 
  VectorXd coeffs;
  CorrelatedWavefunction<Corr, Reference> wave; //reference wavefunction
  

  EOM()
  {
    //wave = CorrelatedWavefunction<Corr, Reference>();
    wave.readWave();
    
    //cout << "JS vars\n";  wave.printVariables();
    // Resize coeffs
    int norbs = Determinant::norbs;
    int numCoeffs = 1 + 2 * norbs * norbs;
    //these are arranged like this: ref, up single excitations, down single excitations
    coeffs = VectorXd::Zero(numCoeffs); //for the sampling wave

    //coeffs order: phi0, singly excited (spin orb index), doubly excited (spin orb pair index)

    if (commrank == 0) {
      auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                              std::ref(generator));

      coeffs(0) = -0.5;
      //coeffs(1) = -0.5;
      for (int i=1; i < numCoeffs; i++) {
        coeffs(i) = 0.05*random();
      }
    }

#ifndef SERIAL
  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    
    char file[5000];
    sprintf(file, "eom.txt");
    ifstream ofile(file);
    if (ofile) {
      for (int i = 0; i < coeffs.size(); i++) {
        ofile >> coeffs(i);
      }
    }
    //cout << "ciCoeffs\n" << coeffs << endl;

    
  }

  Reference& getRef() { return wave.getRef(); }
  Corr& getCorr() { return wave.getCorr(); }

  void initWalker(Walker<Corr, Reference> &walk)  
  {
    walk = Walker<Corr, Reference>(wave.corr, wave.ref);
  }
  
  void initWalker(Walker<Corr, Reference> &walk, Determinant &d) 
  {
    walk = Walker<Corr, Reference>(wave.corr, wave.ref, d);
  }
  
  void getVariables(VectorXd& vars) {
    vars = coeffs;
  }

  void printVariables() {
    cout << "eom coeffs\n" << coeffs << endl;
  }

  void updateVariables(VectorXd& vars) {
    coeffs = vars;
  }

  long getNumVariables() {
    return coeffs.size();
  }
  
  double getOverlapFactor(int i, int a, const Walker<Corr, Reference>& walk, bool doparity) const  
  {
    return 1.;
  }//not used
  
  double getOverlapFactor(int I, int J, int A, int B, const Walker<Corr, Reference>& walk, bool doparity) const 
  {
    return 1.;
  }//not used
  
  //updates from and to by combining with excitation ex
  //used for m -> m'
  void combineExcitations(int ex, std::array<unordered_set<int>, 2> &from, std::array<unordered_set<int>, 2> &to)
  {
    int norbs = Determinant::norbs;
    int I = ex / (2 * norbs), A = ex % (2 * norbs);
    int i = I / 2, a = A / 2;
    bool sz = I%2;
    auto itOrbsi = to[sz].find(i); 
    auto itHolesa = from[sz].find(a); 
    if (itOrbsi == to[sz].end()) from[sz].insert(i);
    else to[sz].erase(itOrbsi);
    if (itHolesa == from[sz].end()) to[sz].insert(a);
    else from[sz].erase(itHolesa);
  }
 
  //generates excitations from the walker due to the eom operators (n -> n')
  //and calculates the ovlpRatios for these excitations
  //returns squareRatio | < n | psi_0 > / < n | psi_s > |^2
  double ovlpRatio(Walker<Corr, Reference>& n, VectorXd& ovlpRation, double &ovlpR) {
    int norbs = Determinant::norbs;
    ovlpR = 0.;
    ovlpRation(0) = 1.;
    ovlpR += coeffs(0);
    for (int sz = 0; sz < 2; sz++) {
      for (int p = 0; p < norbs; p++) {
        //q = p
        int index = 1 + (sz * norbs * norbs) + (p * norbs) + p;
        if (n.d.getocc(p, sz)) ovlpRation(index) = 1.;
        ovlpR += coeffs(index);
        for (int q = 0; q < p; q++) {
          index = 1 + (sz * norbs * norbs) + (p * norbs) + q;
          if (n.d.getocc(p, sz) && (!n.d.getocc(q, sz))) ovlpRation(index) = wave.getOverlapFactor(2*p + sz, 2*q + sz, n, false);
          ovlpR += coeffs(index) * ovlpRation(index);
        }
        for (int q = p+1; q < norbs; q++) {
          index = 1 + (sz * norbs * norbs) + (p * norbs) + q;
          if (n.d.getocc(p, sz) && (!n.d.getocc(q, sz))) ovlpRation(index) = wave.getOverlapFactor(2*p + sz, 2*q + sz, n, false);
          ovlpR += coeffs(index) * ovlpRation(index);
        }
      }
    }
    return 1 / ovlpR / ovlpR;
  }
  
  //generates excitations from the determinant due to the eom operators (m -> m')
  //and calculates ovlpRatios for n -> m' excitations
  //returns < m | psi_s > / < n | psi_0 > 
  double ovlpRatio(Walker<Corr, Reference>& n, Determinant m, std::array<unordered_set<int>, 2> &from, std::array<unordered_set<int>, 2> &to, VectorXd& ovlpRatiom) {
    int norbs = Determinant::norbs;
    double ratio = 0.;
    ovlpRatiom(0) = wave.getOverlapFactor(n, from, to);
    ratio += coeffs(0) * ovlpRatiom(0);
    for (int sz = 0; sz < 2; sz++) {
      for (int p = 0; p < norbs; p++) {
        int index = 1 + sz * norbs * norbs + p * norbs + p;
        if (m.getocc(p, sz)) ovlpRatiom(index) = wave.getOverlapFactor(n, from, to);
        ratio += coeffs(index) * ovlpRatiom(index);
        for (int q = 0; q < p; q++) {
          index = 1 + sz * norbs * norbs + p * norbs + q;
          auto fromNew = from;
          auto toNew = to;
          combineExcitations((2*p +sz) * 2*norbs + (2*q + sz), fromNew, toNew);
          if (m.getocc(p, sz) && !m.getocc(q, sz)) ovlpRatiom(index) = m.parity(q, p, sz) * wave.getOverlapFactor(n, fromNew, toNew);
          ratio += coeffs(index) * ovlpRatiom(index);
        }
        for (int q = p+1; q < norbs; q++) {
          index = 1 + sz * norbs * norbs + p * norbs + q;
          auto fromNew = from;
          auto toNew = to;
          combineExcitations((2*p +sz) * 2*norbs + (2*q + sz), fromNew, toNew);
          if (m.getocc(p, sz) && !m.getocc(q, sz)) ovlpRatiom(index) = m.parity(q, p, sz) * wave.getOverlapFactor(n, fromNew, toNew);
          ratio += coeffs(index) * ovlpRatiom(index);
        }
      }
    }
    return ratio;
  }

  void HamAndOvlp(Walker<Corr, Reference> &n,
                  double &ovlp, MatrixXd &hamSample, MatrixXd &ovlpSample, 
                  workingArray& work) 
  {
    //cout << "\nin HamAndOvlp\n";
    int norbs = Determinant::norbs;
    double dEne = n.d.Energy(I1, I2, coreE);
    double ovlpR = 0.; // < n | psi_s > / < n | psi_0 >
    VectorXd ovlpRation = VectorXd::Zero(coeffs.size()); // < n | pq > / < n | psi_0 >
    VectorXd hamVec = VectorXd::Zero(coeffs.size()); // < n | H | pq > / < n | psi_0 >
    double sampleRatioSquare = ovlpRatio(n, ovlpRation, ovlpR); // | < n | psi_0 > / < n | psi_s > |^2
    ovlpSample = sampleRatioSquare * ovlpRation * ovlpRation.transpose();
    double ovlp0 = wave.Overlap(n); // < n | psi_0 >
    //ovlpSample = ovlpRation * ovlpRation.transpose() * ovlp0 * ovlp0;
    ovlp = ovlpR * ovlp0; // < n | psi_s >
    //cout << "ovlp0  " << ovlp0 << "  ovlps  " << ovlp << endl << endl;
    //cout << "ovlpRation\n" << ovlpRation << endl << endl;
    hamVec += dEne * ovlpRation;
    work.setCounterToZero();
    generateAllScreenedSingleExcitation(n.d, schd.epsilon, schd.screen,
                                        work, false);
    generateAllScreenedDoubleExcitation(n.d, schd.epsilon, schd.screen,
                                        work, false);
   
    //loop over all the screened excitations
    //cout << "\ngenerating m's\n\n";
    for (int i=0; i<work.nExcitations; i++) {
      double tia = work.HijElement[i];
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      double parity = 1.;//ham element parity
      //cout << "ex1  " << ex1 << "  ex2  " << ex2 << endl; 
      std::array<unordered_set<int> , 2> from, to;
      from[0].clear(); to[0].clear();
      from[1].clear(); to[1].clear();
      from[I%2].insert(I/2);
      to[A%2].insert(A/2);
      //cout << endl << "from u  ";
      //copy(from[0].begin(), from[0].end(), ostream_iterator<int>(cout, " "));
      //cout << endl << "from d  ";
      //copy(from[1].begin(), from[1].end(), ostream_iterator<int>(cout, " "));
      //cout << endl << "to u  ";
      //copy(to[0].begin(), to[0].end(), ostream_iterator<int>(cout, " "));
      //cout << endl << "to d  ";
      //copy(to[1].begin(), to[1].end(), ostream_iterator<int>(cout, " "));
      //cout << endl;
      Determinant m = n.d;
      parity *= m.parity(A/2, I/2, I%2);
      //if (ex2 == 0) {
      //  mEne = dEne + n.energyIntermediates[A%2][A/2] - n.energyIntermediates[I%2][I/2] 
      //              - (I2.Direct(I/2, A/2) - I2.Exchange(I/2, A/2));
      //}
      //else {
      m.setocc(I, false);
      m.setocc(A, true);
      if (ex2 != 0) {
        from[J%2].insert(J/2);
        to[B%2].insert(B/2);
        parity *= m.parity(B/2, J/2, J%2);
        m.setocc(J, false);
        m.setocc(B, true);
        //bool sameSpin = (I%2 == J%2);
        //mEne = dEne + n.energyIntermediates[A%2][A/2] - n.energyIntermediates[I%2][I/2]
        //            + n.energyIntermediates[B%2][B/2] - n.energyIntermediates[J%2][J/2]
        //            + I2.Direct(A/2, B/2) - sameSpin * I2.Exchange(A/2, B/2)
        //            + I2.Direct(I/2, J/2) - sameSpin * I2.Exchange(I/2, J/2)
        //            - (I2.Direct(I/2, A/2) - I2.Exchange(I/2, A/2))
        //            - (I2.Direct(J/2, B/2) - I2.Exchange(J/2, B/2))
        //            - (I2.Direct(I/2, B/2) - sameSpin * I2.Exchange(I/2, B/2))
        //            - (I2.Direct(J/2, A/2) - sameSpin * I2.Exchange(J/2, A/2));
      } 
      //cout << "m  " << m << endl;
      VectorXd ovlpRatiom = VectorXd::Zero(coeffs.size()); // < m | pq > / < n | psi_0 >
      double ratiom = ovlpRatio(n, m, from, to, ovlpRatiom); // < m | psi_s > / < n | psi_0 >
      hamVec += parity * tia * ovlpRatiom;
      work.ovlpRatio[i] = ratiom * ovlp0 / ovlp; // < m | psi_s> / < n | psi_s >
      //ham += tia * ovlpRatiom;
      //cout << "hamSample   " << tia * ovlpRatiom << endl;
      //cout << "ovlpRatiom(0)  " << ovlpRatiom(0) <<  "  tia  " << tia << "  parity  " << parity << endl << endl;
    }
    //cout << "hamVec\n" << hamVec << endl << endl;
    hamSample = sampleRatioSquare * ovlpRation * hamVec.transpose();
  }
  
  double optimizeWaveDeterministic(Walker<Corr, Reference>& walk) {

    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    vector<Determinant> allDets;
    generateAllDeterminants(allDets, norbs, nalpha, nbeta);

    workingArray work;
    double overlapTot = 0.; 
    MatrixXd hamSample = VectorXd::Zero(coeffs.size());
    MatrixXd ovlpSample = VectorXd::Zero(coeffs.size());
    MatrixXd ciHam = MatrixXd::Zero(coeffs.size(), coeffs.size());
    MatrixXd sMat = MatrixXd::Zero(coeffs.size(), coeffs.size());// + 1.e-6 * MatrixXd::Identity(coeffs.size(), coeffs.size());
    //w.printVariables();

    for (int i = commrank; i < allDets.size(); i += commsize) {
      wave.initWalker(walk, allDets[i]);
      if (schd.debug) {
        cout << "walker\n" << walk << endl;
      }
      double ovlp = 0.;
      HamAndOvlp(walk, ovlp, hamSample, ovlpSample, work);
      //cout << "ham  " << ham[0] << "  " << ham[1] << "  " << ham[2] << endl;
      //cout << "ovlp  " << ovlp[0] << "  " << ovlp[1] << "  " << ovlp[2] << endl << endl;
      
      overlapTot += ovlp * ovlp;
      ciHam += (ovlp * ovlp) * hamSample;
      sMat += (ovlp * ovlp) * ovlpSample;
      //sMat += ovlpSample;
      hamSample.setZero();
      ovlpSample.setZero();
    }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(overlapTot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ciHam.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sMat.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    double ene0 = 0.;
    if (commrank == 0) {
      ciHam = ciHam / overlapTot;
      sMat = sMat / overlapTot;
      ene0 = ciHam(0, 0) / sMat(0,0);
      //sMat += 1.e-8 * MatrixXd::Identity(coeffs.size(), coeffs.size());
      //cout << "ciHam\n" << ciHam << endl << endl;
      //cout << "sMat\n" << sMat << endl << endl; 
      //GeneralizedEigenSolver<MatrixXd> diag(ciHam, sMat);
      //VectorXd::Index minInd;
      //double minEne = diag.eigenvalues().real().minCoeff(&minInd);
      //coeffs = diag.eigenvectors().col(minInd).real();
      cout << "ref energy   " << fixed << setprecision(5) << ene0 << endl;
      MatrixXd projSMat = sMat.block(1,1,coeffs.size()-1, coeffs.size()-1) - sMat.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1);
      MatrixXd projHam = ciHam.block(1,1,coeffs.size()-1, coeffs.size()-1)- ciHam.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1)
                               - sMat.block(1,0,coeffs.size()-1,1) * ciHam.block(0,1,1,coeffs.size()-1)
                               + ene0 * sMat.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1); 
      GeneralizedEigenSolver<MatrixXd> diag(projHam, projSMat);
      cout << "eigenvalues\n" << diag.eigenvalues() << endl;
      //cout << "eigenvalues\n" << diag.eigenvalues() << endl;
      //cout << "ciHam\n" << ciHam << endl;
      //cout << "sMat\n" << sMat << endl;
      //cout << "coeffs\n" << coeffs << endl;
      //EigenSolver<MatrixXd> ovlpDiag(sMat);
      //VectorXd norms = ovlpDiag.eigenvalues().real();
      //std::vector<int> largeNormIndices;
      //for (int i = 0; i < coeffs.size(); i++) {
      //  if (norms(i) > 1.e-5) {
      //    largeNormIndices.push_back(i);
      //  }
      //}
      //Map<VectorXi> largeNormSlice(&largeNormIndices[0], largeNormIndices.size());
      //VectorXd largeNorms;
      //igl::slice(norms, largeNormSlice, largeNorms);
      //cout << "largeNorms size  " << largeNorms.size() << endl;
      //DiagonalMatrix<double, Dynamic> largeNormInv;
      //largeNormInv.resize(largeNorms.size());
      //largeNormInv.diagonal() = largeNorms.cwiseSqrt().cwiseInverse();
      ////largeNormInv.diagonal() = largeNorms.cwiseInverse();
      //MatrixXd largeBasis;
      //igl::slice(ovlpDiag.eigenvectors().real(), VectorXi::LinSpaced(coeffs.size(), 0, coeffs.size()-1), largeNormSlice, largeBasis);
      //MatrixXd basisChange = largeBasis * largeNormInv;
      //MatrixXd largeHam = basisChange.transpose() * ciHam * basisChange;
      //EigenSolver<MatrixXd> hamDiag(largeHam);
      //cout << "eigenvalues\n" << hamDiag.eigenvalues().real() << endl;
    }
#ifndef SERIAL
  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //MPI_Bcast(&(ene0), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }
  
  double calcPolDeterministic(Walker<Corr, Reference>& walk) {

    int norbs = Determinant::norbs;
    int nalpha = Determinant::nalpha;
    int nbeta = Determinant::nbeta;
    vector<Determinant> allDets;
    generateAllDeterminants(allDets, norbs, nalpha, nbeta);

    workingArray work;
    double overlapTot = 0.; 
    MatrixXd hamSample = VectorXd::Zero(coeffs.size());
    MatrixXd ovlpSample = VectorXd::Zero(coeffs.size());
    MatrixXd ciHam = MatrixXd::Zero(coeffs.size(), coeffs.size());
    MatrixXd sMat = MatrixXd::Zero(coeffs.size(), coeffs.size());// + 1.e-6 * MatrixXd::Identity(coeffs.size(), coeffs.size());
    //w.printVariables();

    for (int i = commrank; i < allDets.size(); i += commsize) {
      wave.initWalker(walk, allDets[i]);
      if (schd.debug) {
        cout << "walker\n" << walk << endl;
      }
      double ovlp = 0.;
      HamAndOvlp(walk, ovlp, hamSample, ovlpSample, work);
      //cout << "ham  " << ham[0] << "  " << ham[1] << "  " << ham[2] << endl;
      //cout << "ovlp  " << ovlp[0] << "  " << ovlp[1] << "  " << ovlp[2] << endl << endl;
      
      overlapTot += ovlp * ovlp;
      ciHam += (ovlp * ovlp) * hamSample;
      sMat += (ovlp * ovlp) * ovlpSample;
      //sMat += ovlpSample;
      hamSample.setZero();
      ovlpSample.setZero();
    }

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, &(overlapTot), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ciHam.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sMat.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    double ene0 = 0.;
    if (commrank == 0) {
      ciHam = ciHam / overlapTot;
      sMat = sMat / overlapTot;
      ene0 = ciHam(0, 0) / sMat(0,0);
      sMat += 1.e-8 * MatrixXd::Identity(coeffs.size(), coeffs.size());
      //cout << "ciHam\n" << ciHam << endl << endl;
      //cout << "sMat\n" << sMat << endl << endl; 
      //GeneralizedEigenSolver<MatrixXd> diag(ciHam, sMat);
      //VectorXd::Index minInd;
      //double minEne = diag.eigenvalues().real().minCoeff(&minInd);
      //coeffs = diag.eigenvectors().col(minInd).real();
      cout << "ref energy   " << fixed << setprecision(5) << ene0 << endl;
      MatrixXd projSMat = sMat.block(1,1,coeffs.size()-1, coeffs.size()-1) - sMat.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1);
      MatrixXd projHam = ciHam.block(1,1,coeffs.size()-1, coeffs.size()-1)- ciHam.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1)
                               - sMat.block(1,0,coeffs.size()-1,1) * ciHam.block(0,1,1,coeffs.size()-1)
                               + ene0 * sMat.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1); 
      //cout << "eigenvalues\n" << diag.eigenvalues() << endl;
      //cout << "ciHam\n" << ciHam << endl;
      //cout << "sMat\n" << sMat << endl;
      //cout << "coeffs\n" << coeffs << endl;
      
      //EigenSolver<MatrixXd> ovlpDiag(sMat);
      //VectorXd norms = ovlpDiag.eigenvalues().real();
      //std::vector<int> largeNormIndices;
      //for (int i = 0; i < coeffs.size(); i++) {
      //  if (norms(i) > 1.e-5) {
      //    largeNormIndices.push_back(i);
      //  }
      //}
      //Map<VectorXi> largeNormSlice(&largeNormIndices[0], largeNormIndices.size());
      //VectorXd largeNorms;
      //igl::slice(norms, largeNormSlice, largeNorms);
      //cout << "largeNorms size  " << largeNorms.size() << endl;
      //DiagonalMatrix<double, Dynamic> largeNormInv;
      //largeNormInv.resize(largeNorms.size());
      //largeNormInv.diagonal() = largeNorms.cwiseSqrt().cwiseInverse();
      ////largeNormInv.diagonal() = largeNorms.cwiseInverse();
      //MatrixXd largeBasis;
      //igl::slice(ovlpDiag.eigenvectors().real(), VectorXi::LinSpaced(coeffs.size(), 0, coeffs.size()-1), largeNormSlice, largeBasis);
      //MatrixXd basisChange = largeBasis * largeNormInv;
      //MatrixXd largeHam = basisChange.transpose() * ciHam * basisChange;
      //VectorXcd stateOnTheSide = basisChange.transpose() * sMat.col(1);
      //for (int i = 0; i < 100; i++) {
      //  complex<double> w (-0.5 + 0.005 * i, 0.001);
      //  MatrixXcd aMat = (w - ene0) * MatrixXcd::Identity(largeNorms.size(), largeNorms.size()) + largeHam;
      //  Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> lin(aMat);
      //  VectorXcd corr = lin.solve(stateOnTheSide);
      //  cout << w << "    " << (stateOnTheSide.adjoint() * corr).imag() << endl;
      //}

      VectorXcd stateOnTheSide = VectorXcd::Zero(coeffs.size() - 1);
      stateOnTheSide.real() = projSMat.col(0);
      for (int i = 0; i < 100; i++) {
        complex<double> w (-0.5 + 0.005 * i, 0.001);
        MatrixXcd aMat = (w - ene0) * projSMat + projHam;
        //Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> lin(aMat);
        //Eigen::ConjugateGradient<Eigen::MatrixXcd> lin(aMat);
        Eigen::BiCGSTAB<Eigen::MatrixXcd> lin(aMat);
        VectorXcd corr = lin.solve(stateOnTheSide);
        cout << w << "    " << (stateOnTheSide.adjoint() * corr).imag() << endl;
      }
    }
  }

  void generalizedJacobiDavidson(const Eigen::MatrixXd &H, const Eigen::MatrixXd &S, double &lambda, Eigen::VectorXd &v)
  {
      int dim = H.rows(); //dimension of problem
      int restart = std::max((int) (0.1 * dim), 30); //20 - 30
      int q = std::max((int) (0.04 * dim), 5); //5 - 10
      Eigen::MatrixXd V, HV, SV;  //matrices storing action of vector on sample space
      //Eigen::VectorXd z = Eigen::VectorXd::Random(dim);
      //Eigen::VectorXd z = Eigen::VectorXd::Unit(dim, 0);
      Eigen::VectorXd z = Eigen::VectorXd::Constant(dim, 0.01);
      while (1)
      {
          int m = V.cols(); //number of vectors in subspace at current iteration
  
          //modified grahm schmidt to orthogonalize sample space with respect to overlap matrix
          Eigen::VectorXd Sz = S * z;
          for (int i = 0; i < m; i++)
          {
              double alpha = V.col(i).adjoint() * Sz;
              z = z - alpha * V.col(i);
          }
  
          //normalize z after orthogonalization and calculate action of matrices on z
          Sz = S * z;
          double beta = std::sqrt(z.adjoint() * Sz);
          Eigen::VectorXd v_m = z / beta;
          Eigen::VectorXd Hv_m = H * v_m;
          Eigen::VectorXd Sv_m = S * v_m;
          //store new vector
          V.conservativeResize(H.rows(), m + 1);
          HV.conservativeResize(H.rows(), m + 1);
          SV.conservativeResize(H.rows(), m + 1);
          V.col(m) = v_m;
          HV.col(m) = Hv_m;
          SV.col(m) = Sv_m;
  
          //solve eigenproblem in subspace
          Eigen::MatrixXd A = V.adjoint() * HV;
          Eigen::MatrixXd B = V.adjoint() * SV;
          Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(A, B);
          //if sample space size is too large, restart subspace
          if (m == restart)
          {
              Eigen::MatrixXd N = es.eigenvectors().block(0, 0, A.rows(), q);
              V = V * N;
              HV = HV * N;
              SV = SV * N;
              A = V.adjoint() * HV;
              B = V.adjoint() * SV;
              es.compute(A, B);
          }
          Eigen::VectorXd s = es.eigenvectors().col(0);
          double theta = es.eigenvalues()(0);
          //cout << "theta: " << theta << endl;
  
          //transform vector into original space
          Eigen::VectorXd u = V * s;
          //calculate residue vector
          Eigen::VectorXd u_H = HV * s;
          Eigen::VectorXd u_S = SV * s;
          Eigen::VectorXd r = u_H - theta * u_S;
  
          if (r.squaredNorm() < 1.e-6)
          {
              lambda = theta;
              v = u;
              break;
          }
          
          Eigen::MatrixXd X = Eigen::MatrixXd::Identity(u.rows(), u.rows());
          Eigen::MatrixXd Xleft = X - S * u * u.adjoint();
          Eigen::MatrixXd Xright = X - u * u.adjoint() * S;
          X = Xleft * (H - theta * S) * Xright;
          Eigen::ColPivHouseholderQR<Eigen::MatrixXd> dec(X);
          z = dec.solve(-r);
      }
  }

  void optimizeWaveCT(Walker<Corr, Reference>& walk) {

    //add noise to avoid zero coeffs
//    if (commrank == 0) {
//      cout << "starting sampling at " << setprecision(4) << getTime() - startofCalc << endl; 
//      auto random = std::bind(std::uniform_real_distribution<double>(0., 1.e-8), std::ref(generator));
//      for (int i=0; i < coeffs.size(); i++) {
//        if (coeffs(i) == 0) coeffs(i) = random();
//      }
//    }
//
//#ifndef SERIAL
//  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
//#endif

    //sampling
    int norbs = Determinant::norbs;
    auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                            std::ref(generator));
    workingArray work;
    double ovlp = 0.; 
    MatrixXd hamSample = VectorXd::Zero(coeffs.size());
    MatrixXd ovlpSample = VectorXd::Zero(coeffs.size());
    MatrixXd ciHam = MatrixXd::Zero(coeffs.size(), coeffs.size());
    MatrixXd sMat = MatrixXd::Zero(coeffs.size(), coeffs.size());
    //cout << "walker\n" << walk << endl;
    HamAndOvlp(walk, ovlp, hamSample, ovlpSample, work);

    int iter = 0;
    double cumdeltaT = 0;
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
      sMat *= (1 - ratio);
      sMat += ratio * ovlpSample;
      ciHam *= (1 - ratio);
      ciHam += ratio * hamSample;
      walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
      //cout << "walker\n" << walk << endl;
      hamSample.setZero();
      ovlpSample.setZero();
      HamAndOvlp(walk, ovlp, hamSample, ovlpSample, work);
      iter++;
      if (commrank == 0 && iter % printMod == 1) cout << "iter  " << iter << "  t  " << getTime() - startofCalc << endl; 
    }
    
    sMat *= cumdeltaT;
    ciHam *= cumdeltaT;

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, sMat.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ciHam.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(cumdeltaT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    if (commrank == 0) {
      sMat /= cumdeltaT;
      ciHam /= cumdeltaT;
      //ciHam = (ciHam + ciHam.transpose().eval()) / 2;
      double ene0 = ciHam(0, 0) / sMat(0, 0);
      cout << "ref energy   " << setprecision(12) << ciHam(0, 0) / sMat(0, 0) << endl;
      //sMat += 1.e-7 * MatrixXd::Identity(coeffs.size(), coeffs.size());
      //cout << "ciHam\n" << ciHam << endl << endl;
      //cout << "sMat\n" << sMat << endl << endl; 
      MatrixXd projSMat = sMat.block(1,1,coeffs.size()-1, coeffs.size()-1) - sMat.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1);
      MatrixXd projHam = ciHam.block(1,1,coeffs.size()-1, coeffs.size()-1)- ciHam.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1)
                               - sMat.block(1,0,coeffs.size()-1,1) * ciHam.block(0,1,1,coeffs.size()-1)
                               + ene0 * sMat.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1); 
      EigenSolver<MatrixXd> ovlpDiag(projSMat);
      VectorXd norms = ovlpDiag.eigenvalues().real();
      std::vector<int> largeNormIndices;
      for (int i = 0; i < coeffs.size()-1; i++) {
        if (norms(i) > schd.overlapCutoff) {
          largeNormIndices.push_back(i);
        }
      }
      Map<VectorXi> largeNormSlice(&largeNormIndices[0], largeNormIndices.size());
      VectorXd largeNorms;
      igl::slice(norms, largeNormSlice, largeNorms);
      cout << "largeNorms size  " << largeNorms.size() << endl;
      DiagonalMatrix<double, Dynamic> largeNormInv;
      largeNormInv.resize(largeNorms.size());
      largeNormInv.diagonal() = largeNorms.cwiseSqrt().cwiseInverse();
      //largeNormInv.diagonal() = largeNorms.cwiseInverse();
      MatrixXd largeBasis;
      igl::slice(ovlpDiag.eigenvectors().real(), VectorXi::LinSpaced(coeffs.size()-1, 0, coeffs.size()-2), largeNormSlice, largeBasis);
      MatrixXd basisChange = largeBasis * largeNormInv;
      MatrixXd largeHam = basisChange.transpose() * projHam * basisChange;
      EigenSolver<MatrixXd> hamDiag(largeHam);
      cout << "eigenvalues\n" << hamDiag.eigenvalues().real() << endl;
      //GeneralizedEigenSolver<MatrixXd> diag(ciHam, sMat);
      //VectorXd::Index minInd;
      //double minEne = diag.eigenvalues().real().minCoeff(&minInd);
      //double ene0 = 0.; VectorXd gs = VectorXd::Zero(coeffs.size());
      //generalizedJacobiDavidson(ciHam, sMat, ene0, gs);
      //cout << "ene0  " << ene0 << endl;
      //coeffs = diag.eigenvectors().col(minInd).real();
      //cout << "ciHam\n" << ciHam << endl << endl;
      //cout << "sMat\n" << sMat << endl << endl;
      //cout << "eigenvalues\n" << diag.eigenvalues() << endl;
    }
  }
  
  void calcPolCT(Walker<Corr, Reference>& walk) {

    //add noise to avoid zero coeffs
//    if (commrank == 0) {
//      cout << "starting sampling at " << setprecision(4) << getTime() - startofCalc << endl; 
//      auto random = std::bind(std::uniform_real_distribution<double>(0., 1.e-8), std::ref(generator));
//      for (int i=0; i < coeffs.size(); i++) {
//        if (coeffs(i) == 0) coeffs(i) = random();
//      }
//    }
//
//#ifndef SERIAL
//  MPI_Bcast(coeffs.data(), coeffs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
//#endif

    //sampling
    int norbs = Determinant::norbs;
    auto random = std::bind(std::uniform_real_distribution<double>(0, 1),
                            std::ref(generator));
    workingArray work;
    double ovlp = 0.; 
    MatrixXd hamSample = VectorXd::Zero(coeffs.size());
    MatrixXd ovlpSample = VectorXd::Zero(coeffs.size());
    MatrixXd ciHam = MatrixXd::Zero(coeffs.size(), coeffs.size());
    MatrixXd sMat = MatrixXd::Zero(coeffs.size(), coeffs.size());
    //cout << "walker\n" << walk << endl;
    HamAndOvlp(walk, ovlp, hamSample, ovlpSample, work);

    int iter = 0;
    double cumdeltaT = 0;
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
      sMat *= (1 - ratio);
      sMat += ratio * ovlpSample;
      ciHam *= (1 - ratio);
      ciHam += ratio * hamSample;
      walk.updateWalker(wave.getRef(), wave.getCorr(), work.excitation1[nextDet], work.excitation2[nextDet]);
      //cout << "walker\n" << walk << endl;
      hamSample.setZero();
      ovlpSample.setZero();
      HamAndOvlp(walk, ovlp, hamSample, ovlpSample, work);
      iter++;
      if (commrank == 0 && iter % printMod == 1) cout << "iter  " << iter << "  t  " << getTime() - startofCalc << endl; 
    }
    
    sMat *= cumdeltaT;
    ciHam *= cumdeltaT;

#ifndef SERIAL
  MPI_Allreduce(MPI_IN_PLACE, sMat.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, ciHam.data(), coeffs.size() * coeffs.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &(cumdeltaT), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    
    if (commrank == 0) {
      sMat /= cumdeltaT;
      ciHam /= cumdeltaT;
      //ciHam = (ciHam + ciHam.transpose().eval()) / 2;
      double ene0 = ciHam(0, 0) / sMat(0, 0);
      cout << "ref energy   " << setprecision(12) << ciHam(0, 0) / sMat(0, 0) << endl;
      MatrixXd projSMat = sMat.block(1,1,coeffs.size()-1, coeffs.size()-1) - sMat.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1);
      MatrixXd projHam = ciHam.block(1,1,coeffs.size()-1, coeffs.size()-1)- ciHam.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1)
                               - sMat.block(1,0,coeffs.size()-1,1) * ciHam.block(0,1,1,coeffs.size()-1)
                               + ene0 * sMat.block(1,0,coeffs.size()-1,1) * sMat.block(0,1,1,coeffs.size()-1); 
      //sMat += 1.e-7 * MatrixXd::Identity(coeffs.size(), coeffs.size());
      //cout << "ciHam\n" << ciHam << endl << endl;
      //cout << "sMat\n" << sMat << endl << endl; 
      
      //EigenSolver<MatrixXd> ovlpDiag(sMat);
      //VectorXd norms = ovlpDiag.eigenvalues().real();
      //std::vector<int> largeNormIndices;
      //for (int i = 0; i < coeffs.size(); i++) {
      //  if (norms(i) > schd.overlapCutoff) {
      //    largeNormIndices.push_back(i);
      //  }
      //}
      //Map<VectorXi> largeNormSlice(&largeNormIndices[0], largeNormIndices.size());
      //VectorXd largeNorms;
      //igl::slice(norms, largeNormSlice, largeNorms);
      //cout << "largeNorms size  " << largeNorms.size() << endl;
      //DiagonalMatrix<double, Dynamic> largeNormInv;
      //largeNormInv.resize(largeNorms.size());
      //largeNormInv.diagonal() = largeNorms.cwiseSqrt().cwiseInverse();
      ////largeNormInv.diagonal() = largeNorms.cwiseInverse();
      //MatrixXd largeBasis;
      //igl::slice(ovlpDiag.eigenvectors().real(), VectorXi::LinSpaced(coeffs.size(), 0, coeffs.size()-1), largeNormSlice, largeBasis);
      //MatrixXd basisChange = largeBasis * largeNormInv;
      //MatrixXd largeHam = basisChange.transpose() * ciHam * basisChange;
      //VectorXcd stateOnTheSide = basisChange.transpose() * sMat.col(2);
      //for (int i = 0; i < 100; i++) {
      //  complex<double> w (-0.5 + 0.005 * i, 0.001);
      //  MatrixXcd aMat = (w - ene0) * MatrixXcd::Identity(largeNorms.size(), largeNorms.size()) + largeHam;
      //  Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> lin(aMat);
      //  VectorXcd corr = lin.solve(stateOnTheSide);
      //  cout << w << "    " << (stateOnTheSide.adjoint() * corr).imag() << endl;
      //}
      
      VectorXcd stateOnTheSide = VectorXcd::Zero(coeffs.size()-1);
      stateOnTheSide.real() = projSMat.col(0);
      for (int i = 0; i < 100; i++) {
        complex<double> w (-0.5 + 0.005 * i, 0.001);
        MatrixXcd aMat = (w - ene0) * projSMat + projHam;
        //Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> lin(aMat);
        //Eigen::ConjugateGradient<Eigen::MatrixXcd> lin(aMat);
        Eigen::BiCGSTAB<Eigen::MatrixXcd> lin(aMat);
        VectorXcd corr = lin.solve(stateOnTheSide);
        cout << w << "    " << (stateOnTheSide.adjoint() * corr).imag() << endl;
      }
    }
  }
  
  string getfileName() const {
    return "eom"+wave.getfileName();
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
