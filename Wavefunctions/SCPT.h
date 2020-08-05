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

#include <unordered_map>
#include <iomanip>
#include "Determinants.h"
#include "workingArray.h"
#include <boost/filesystem.hpp>

#ifndef SERIAL
#include "mpi.h"
#endif

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
      ar & wave
      ar & coeffs
      ar & moEne;
    }

 public:
  Eigen::VectorXd coeffs;
  Eigen::VectorXd moEne;
  Wfn wave; // reference wavefunction
  workingArray morework;

  double ovlp_current;
  
  static const int NUM_EXCIT_CLASSES = 9;
  // the number of coefficients in each excitation class
  int numCoeffsPerClass[NUM_EXCIT_CLASSES];
  // the cumulative sum of numCoeffsPerClass
  int cumNumCoeffs[NUM_EXCIT_CLASSES];
  // the total number of strongly contracted states (including the CASCI space itself)
  int numCoeffs;
  // a list of the excitation classes being considered stochastically
  std::array<bool, NUM_EXCIT_CLASSES> classesUsed = { false };
  // a list of the excitation classes being considered deterministically
  std::array<bool, NUM_EXCIT_CLASSES> classesUsedDeterm = { false };
  // a list of classes for which the perturber norms are calculated deterministically
  std::array<bool, NUM_EXCIT_CLASSES> normsDeterm = { false };

  std::unordered_map<std::array<int,3>, int, boost::hash<std::array<int,3>> > class_1h2p_ind;
  std::unordered_map<std::array<int,3>, int, boost::hash<std::array<int,3>> > class_2h1p_ind;
  std::unordered_map<std::array<int,4>, int, boost::hash<std::array<int,4>> > class_2h2p_ind;

  // the names of each of the 9 classes
  string classNames[NUM_EXCIT_CLASSES] = {"CASCI", "AAAV", "AAVV", "CAAA", "CAAV", "CAVV", "CCAA", "CCAV", "CCVV"};
  string classNames2[NUM_EXCIT_CLASSES] = {"CASCI", "   V", "  VV", "   C", "  CV", " CVV", "  CC", " CCV", "CCVV"};


  SCPT();

  void createClassIndMap(int& numStates_1h2p, int& numStates_2h1p, int& numStates_2h2p);

  typename Wfn::ReferenceType& getRef();
  typename Wfn::CorrType& getCorr();

  template<typename Walker>
  void initWalker(Walker& walk);
  
  template<typename Walker>
  void initWalker(Walker& walk, Determinant& d);
  
  //void initWalker(Walker& walk);

  void getVariables(Eigen::VectorXd& vars);

  void printVariables();

  void updateVariables(Eigen::VectorXd& vars);

  long getNumVariables();

  template<typename Walker>
  int coeffsIndex(Walker& walk);

  // This perform the inverse of the coeffsIndex function: given the
  // index of a perturber, return the external (non-active) orbtials
  // involved. This only works for the main perturber types used -
  // V, VV, C, CV, CC. These only have one or two external orbitals,
  // which this function will return.
  void getOrbsFromIndex(const int index, int& i, int& j);

  // Take two orbital indices i and j, and convert them to a string.
  // This is intended for use with two orbital obtained from the
  // getOrbsFromIndex function, which are then to be output to
  // pt2_energies files.
  string formatOrbString(const int i, const int j);
  
  template<typename Walker>
  double getOverlapFactor(int i, int a, const Walker& walk, bool doparity) const;
  
  template<typename Walker>
  double getOverlapFactor(int I, int J, int A, int B, const Walker& walk, bool doparity) const;

  template<typename Walker>
  bool checkWalkerExcitationClass(Walker &walk);
  
  // ham is a sample of the diagonal element of the Dyall ham
  template<typename Walker>
  void HamAndOvlp(Walker &walk,
                  double &ovlp, double &locEne, double &ham, double &norm, int coeffsIndex, 
                  workingArray& work, bool fillExcitations=true);
  
  // ham is a sample of the diagonal element of the Dyall ham
  template<typename Walker>
  void FastHamAndOvlp(Walker &walk, double &ovlp, double &ham, workingArray& work, bool fillExcitations=true);

  template<typename Walker>
  void HamAndSCNorms(Walker &walk, double &ovlp, double &ham, Eigen::VectorXd &normSamples,
                     vector<Determinant>& initDets, vector<double>& largestCoeffs,
                     workingArray& work, bool calcExtraNorms);

  template<typename Walker>
  void AddSCNormsContrib(Walker &walk, double &ovlp, double &ham, Eigen::VectorXd &normSamples,
                         vector<Determinant>& initDets, vector<double>& largestCoeffs,
                         workingArray& work, bool calcExtraNorms, int& ex1, int& ex2,
                         double& tia, size_t& nExcitationsCASCI);

  // this is a version of HamAndSCNorms, optimized for the case where only
  // classes AAAV (class 1) and CAAA (class 3) are needed
  template<typename Walker>
  void HamAndSCNormsCAAA_AAAV(Walker &walk, double &ovlp, double &ham, Eigen::VectorXd &normSamples,
                     vector<Determinant>& initDets, vector<double>& largestCoeffs,
                     workingArray& work, bool calcExtraNorms);


  template<typename Walker>
  double doNEVPT2_CT(Walker& walk);

  // Output the header for the "norms" file, which will output the norms of
  // the strongly contracted (SC) states, divided by the norm of the CASCI
  // state (squared)
  double outputNormFileHeader(FILE* out_norms);

  // Create directories where the norm files will be stored
  void createDirForSCNorms();

  // Create directories where the init_dets files will be stored
  void createDirForInitDets();

  // Create directories where the norm files will be stored
  void createDirForNormsBinary();

  // Print norms to output files.
  // If determClasses is true, only print the norms from classes where the
  // norms are being found exactly.
  // Otherwise, print the norms calculated stochastically, summed up until
  // the current iteration. Also print the current estimate of the the
  // CASCI energy, and the residence time.
  void printSCNorms(int& iter, double& deltaT_Tot, double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, bool determClasses);

  void readStochNorms(double& deltaT_Tot, double& energyCAS_Tot, Eigen::VectorXd& norms_Tot);

  void readDetermNorms(Eigen::VectorXd& norms_Tot);

  // Print initial determinants to output files
  // We only need to print out the occupations in the active spaces
  // The occupations of the core and virtual orbitals are determined
  // from the label of the SC state, which is fixed by the deterministic
  // ordering (the same as used in coeffsIndex).
  void printInitDets(vector<Determinant>& initDets, vector<double>& largestCoeffs);

  // read determinants in to the initDets array from previously output files
  void readInitDets(vector<Determinant>& initDets, vector<double>& largestCoeffs);

  // From a given line of an output file, containing only the occupations of
  // orbitals within the active space, construct the corresponding determinant
  // (with all core obritals occupied, all virtual orbitals unocuppied).
  // This is specifically used by for readInitDets
  void readDetActive(string& line, Determinant& det, double& coeff);

  void printNormDataBinary(vector<Determinant>& initDets, vector<double>& largestCoeffs,
                           double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, double& deltaT_Tot);

  void readNormDataBinary(vector<Determinant>& initDets, vector<double>& largestCoeffs,
                          double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, double& deltaT_Tot, bool readDeltaT);

  void readNormDataText(vector<Determinant>& initDets, vector<double>& largestCoeffs,
                        double& energyCAS_Tot, Eigen::VectorXd& norms_Tot);
  
  template<typename Walker>
  double doNEVPT2_CT_Efficient(Walker& walk);

  template<typename Walker>
  double sampleSCEnergies(Walker& walk, vector<Determinant>& initDets, vector<double>& largestCoeffs,
                          double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, workingArray& work);

  template<typename Walker>
  void doSCEnergyCTMC(Walker& walk, workingArray& work, double& final_ham, double& var, int& samplingIters);


  // Estimate the variance of the weighted mean used to estimate E_l^k
  double SCEnergyVar(vector<double>& x, vector<double>& w);

  template<typename Walker>
  double doSCEnergyCTMCPrint(Walker& walk, workingArray& work, int sampleIter, int nWalk);


  // Wrapper function for calling doSCEnergyCTMC, which estimates E_l^k,
  // given the appropriate input information, and then to print this info
  // to the provded pt2_out file.
  // This is designed to be called by sampleAllSCEnergies.
  template<typename Walker>
  double SCEnergyWrapper(Walker& walk, int iter, FILE * pt2_out, Determinant& det,
                         double& energyCAS_Tot, double norm, int orbi, int orbj,
                         bool exactCalc, bool exactRead, double& SCHam, workingArray& work);


  // Loop over *all* S_l^k subspaces (for the classes AAAV, AAVV, CAAA,
  // CAAV and CCAA) for which the calculated norm is above the threshold,
  // and sample E_l^k for each. The final PT2 energy is then output as a
  // sum over all of these spaces.
  template<typename Walker>
  double sampleAllSCEnergies(Walker& walk, vector<Determinant>& initDets, vector<double>& largestCoeffs,
                             double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, workingArray& work);


  // Loop over *all* S_l^k subspaces (for the classes AAAV, AAVV, CAAA,
  // CAAV and CCAA), and either exactly calculate of read in E_l^k for each.
  // The final PT2 energy is then output as a sum over all of these spaces.
  //
  // The difference between this and sampleAllSCEnergies is that *all*
  // S_l^k are considered, even if the norm was calculated as zero, if run
  // in write mode, and all E_l^k are then written out. If run in read mode,
  // then all E_l^k are read in from this file instead, for quick calculation.
  template<typename Walker>
  double calcAllSCEnergiesExact(Walker& walk, vector<Determinant>& initDets, vector<double>& largestCoeffs,
                                double& energyCAS_Tot, Eigen::VectorXd& norms_Tot, workingArray& work);

  template<typename Walker>
  double doSCEnergyCTMCSync(Walker& walk, int& ind, workingArray& work, string& outputFile);

  template<typename Walker>
  void doSCEnergyExact(Walker& walk, workingArray& work, double& SCHam, double& SCHamVar, int& samplingIters);

  Determinant generateInitDet(int orb1, int orb2);

  template<typename Walker>
  double compareStochPerturberEnergy(Walker& walk, int orb1, int orb2, double CASEnergy, int nsamples);

  template<typename Walker>
  double doNEVPT2_Deterministic(Walker& walk);

  double get_ccvv_energy();

  void readSpinRDM(Eigen::MatrixXd& oneRDM, Eigen::MatrixXd& twoRDM);

  void calc_AAVV_NormsFromRDMs(Eigen::MatrixXd& twoRDM, Eigen::VectorXd& norms);
  void calc_CAAV_NormsFromRDMs(Eigen::MatrixXd& oneRDM, Eigen::MatrixXd& twoRDM, Eigen::VectorXd& norms);
  void calc_CCAA_NormsFromRDMs(Eigen::MatrixXd& oneRDM, Eigen::MatrixXd& twoRDM, Eigen::VectorXd& norms);

  string getfileName() const;

  void writeWave();
  void readWave();

};

// This is a wrapper function which is called during initialization.
// This is where the main NEVPT2 functions are called from.
void initNEVPT_Wrapper();

#endif
