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
#ifndef ResonatingWavefunction_HEADER_H
#define ResonatingWavefunction_HEADER_H
#include "CorrelatedWavefunction.h"
#include "ResonatingWalker.h"

/**
 This is a linear combination of Jastrow Slater correlated wave functions.
 Only the last element of the list is optimized
 */
struct ResonatingWavefunction {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & waveVec
      & corr
      & ref;
  }

 public:
  
  vector<CorrelatedWavefunction<Jastrow, Slater>> waveVec; 
  // these are for convenience in passing to updateWalker, the information is already present in waveVec
  vector<Jastrow> corr;
  vector<Slater> ref;

  // default constructor
  ResonatingWavefunction() {
    try {// if a ResonatingWavefunction is present, add a new CorrelatedWavefunction by reading hf.txt, (and possibly Jastrow.txt)
      readWave();
      if (waveVec.size() > schd.numResonants - 1) {
        waveVec.resize(schd.numResonants - 1);
        corr.resize(schd.numResonants - 1);
        ref.resize(schd.numResonants - 1);
      }
      waveVec.push_back(CorrelatedWavefunction<Jastrow, Slater>());
      //VectorXd v = VectorXd::Zero(waveVec[waveVec.size() - 2].getNumVariables());
      //waveVec[waveVec.size() - 2].getVariables(v);
      //waveVec[waveVec.size() - 1].updateVariables(v);
      //v.resize(0);
      //waveVec[waveVec.size() - 1].corr = waveVec[waveVec.size() - 2].corr;
      //waveVec[waveVec.size() - 1].corr.addNoise();
      corr.push_back(waveVec[waveVec.size() - 1].corr);
      ref.push_back(waveVec[waveVec.size() - 1].ref);
      if (!schd.restart) {
        assert(waveVec.size() == schd.numResonants);
        if (commrank == 0) {
          cout << "Numer of resonants: " << waveVec.size() << endl;
        }
      }
    }
    catch (const boost::archive::archive_exception &e) {
      try {// if a CorrelatedWavefunction is present, add it, and add another by reading hf.txt, (and possibly Jastrow.txt)
        CorrelatedWavefunction<Jastrow, Slater> wave0;
        wave0.readWave();
        waveVec.push_back(wave0);
        waveVec.push_back(CorrelatedWavefunction<Jastrow, Slater>());
        VectorXd v = VectorXd::Zero(waveVec[0].getNumVariables());
        waveVec[0].getVariables(v);
        waveVec[1].updateVariables(v);
        v.resize(0);
        corr.push_back(waveVec[0].corr);
        ref.push_back(waveVec[0].ref);
        corr.push_back(waveVec[1].corr);
        ref.push_back(waveVec[1].ref);
        assert(waveVec.size() == 2);
        if (commrank == 0) cout << "Number of resonants: 2\n";
      }
      catch (const boost::archive::archive_exception &e) {// if no wave function files are present, read a single CorrelatedWavefunction
        waveVec.push_back(CorrelatedWavefunction<Jastrow, Slater>());
        corr.push_back(waveVec[0].corr);
        ref.push_back(waveVec[0].ref);
        assert(waveVec.size() == 1);
        if (commrank == 0) cout << "Number of resonants: 1\n";
      }
    }
  };
  
  vector<Slater>& getRef() { return ref; }
  vector<Jastrow>& getCorr() { return corr; }

  // used at the start of each sampling run
  void initWalker(ResonatingWalker &walk)  
  {
    walk = ResonatingWalker(corr, ref);
  }
  
  // used in deterministic calculations
  void initWalker(ResonatingWalker &walk, Determinant &d) 
  {
    walk = ResonatingWalker(corr, ref, d);
  }
  
  // used in rdm calculations
  double Overlap(const ResonatingWalker &walk) const 
  {
    double overlap = 0.;
    for (int i = 0; i < waveVec.size(); i++) {
      overlap += waveVec[i].Overlap(walk.walkerVec[i]);
    }
    return overlap;
  }
 
  // used in HamAndOvlp below
  double Overlap(const ResonatingWalker &walk, vector<double> &overlaps) const 
  {
    overlaps.resize(waveVec.size(), 0.);
    double totalOverlap = 0.;
    for (int i = 0; i < waveVec.size(); i++) {
      overlaps[i] = waveVec[i].Overlap(walk.walkerVec[i]);
      totalOverlap += overlaps[i];
    }
    return totalOverlap;
  }

  // used in rdm calculations
  double getOverlapFactor(int i, int a, const ResonatingWalker& walk, bool doparity) const  
  {
    vector<double> overlaps;
    double totalOverlap = Overlap(walk, overlaps);
    double numerator = 0.;
    for (int n = 0; n < waveVec.size(); n++) {
      numerator += waveVec[n].getOverlapFactor(i, a, walk.walkerVec[n], doparity) * overlaps[n];
    }
    return numerator / totalOverlap;
  }

  // used in rdm calculations
  double getOverlapFactor(int I, int J, int A, int B, const ResonatingWalker& walk, bool doparity) const  
  {
    if (J == 0 && B == 0) return getOverlapFactor(I, A, walk, doparity);
    vector<double> overlaps;
    double totalOverlap = Overlap(walk, overlaps);
    double numerator = 0.;
    for (int n = 0; n < waveVec.size(); n++) {
      numerator += waveVec[n].getOverlapFactor(I, J, A, B, walk.walkerVec[n], doparity) * overlaps[n];
    }
    return numerator / totalOverlap;
  }
  
  double getOverlapFactor(int i, int a, const ResonatingWalker& walk, vector<double>& overlaps, double& totalOverlap, bool doparity) const  
  {
    double numerator = 0.;
    for (int n = 0; n < waveVec.size(); n++) {
      numerator += waveVec[n].getOverlapFactor(i, a, walk.walkerVec[n], doparity) * overlaps[n];
    }
    return numerator / totalOverlap;
  }

  // used in HamAndOvlp below
  double getOverlapFactor(int I, int J, int A, int B, const ResonatingWalker& walk, vector<double>& overlaps, double& totalOverlap, bool doparity) const  
  {
    if (J == 0 && B == 0) return getOverlapFactor(I, A, walk, overlaps, totalOverlap, doparity);
    double numerator = 0.;
    for (int n = 0; n < waveVec.size(); n++) {
      numerator += waveVec[n].getOverlapFactor(I, J, A, B, walk.walkerVec[n], doparity) * overlaps[n];
    }
    return numerator / totalOverlap;
  }
  
  // gradient overlap ratio, used during sampling
  // just calls OverlapWithGradient on the last wave function
  void OverlapWithGradient(const ResonatingWalker &walk,
                           double &factor,
                           Eigen::VectorXd &grad) const
  {
    vector<double> overlaps;
    double totalOverlap = Overlap(walk, overlaps);
    size_t index = 0;
    for (int i = 0; i < waveVec.size(); i++) {
      size_t numVars_i = waveVec[i].getNumVariables();
      VectorXd grad_i = VectorXd::Zero(numVars_i);
      waveVec[i].OverlapWithGradient(walk.walkerVec[i], factor, grad_i);
      grad.segment(index, numVars_i) = grad_i * overlaps[i] / totalOverlap;
      index += numVars_i;
    }
  }

  void printVariables() const
  {
    for (int i = 0; i < waveVec.size(); i++) {
      cout << "Wave " << i << endl << endl;
      waveVec[i].printVariables();
    }
  }

  // update after vmc optimization
  // updates the wave function helpers as well
  void updateVariables(Eigen::VectorXd &v) 
  {
    size_t index = 0;
    ref.resize(0);
    corr.resize(0);
    for (int i = 0; i < waveVec.size(); i++) {
      size_t numVars_i = waveVec[i].getNumVariables();
      VectorXd v_i = v.segment(index, numVars_i);
      waveVec[i].updateVariables(v_i);
      ref.push_back(waveVec[i].ref);
      corr.push_back(waveVec[i].corr);
      index += numVars_i;
    }
  }

  void getVariables(Eigen::VectorXd &v) const
  {
    v = VectorXd::Zero(getNumVariables());
    size_t index = 0;
    for (int i = 0; i < waveVec.size(); i++) {
      size_t numVars_i = waveVec[i].getNumVariables();
      VectorXd v_i = VectorXd::Zero(numVars_i);
      waveVec[i].getVariables(v_i);
      v.segment(index, numVars_i) = v_i;
      index += numVars_i;
    }
  }

  long getNumVariables() const
  {
    long numVariables = 0;
    for (int i = 0; i < waveVec.size(); i++) numVariables += waveVec[i].getNumVariables();
    return numVariables;
  }

  string getfileName() const {
    return "ResonatingWavefunction";
  }
  
  void writeWave() const
  {
    if (commrank == 0)
    {
      char file[5000];
      //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
      sprintf(file, (getfileName()+".bkp").c_str() );
      std::ofstream outfs(file, std::ios::binary);
      boost::archive::binary_oarchive save(outfs);
      save << *this;
      outfs.close();
    }
  }

  void readWave()
  {
    //if (commrank == 0)
    //{
      char file[5000];
      //sprintf (file, "wave.bkp" , schd.prefix[0].c_str() );
      sprintf(file, (getfileName()+".bkp").c_str() );
      std::ifstream infs(file, std::ios::binary);
      boost::archive::binary_iarchive load(infs);
      load >> *this;
      infs.close();
    //}
#ifndef SERIAL
    //boost::mpi::communicator world;
    //boost::mpi::broadcast(world, *this, 0);
#endif
  }


  // calculates local energy and overlap
  // used directly during sampling
  void HamAndOvlp(const ResonatingWalker &walk,
                  double &ovlp, double &ham, 
                  workingArray& work, bool fillExcitations=true) const
  {
    int norbs = Determinant::norbs;

    vector<double> overlaps;
    ovlp = Overlap(walk, overlaps);
    if (schd.debug) {
      cout << "overlaps\n";
      for (int i = 0; i <overlaps.size(); i++) cout << overlaps[i] << "  ";
      cout << endl;
    }
    ham = walk.walkerVec[0].d.Energy(I1, I2, coreE); 

    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.walkerVec[0].d, schd.epsilon, schd.screen,
                                        work, false);  
    generateAllScreenedDoubleExcitation(walk.walkerVec[0].d, schd.epsilon, schd.screen,
                                        work, false);  
  
    //loop over all the screened excitations
    if (schd.debug) {
      cout << "eloc excitations" << endl;
      cout << "phi0  d.energy" << ham << endl;
    }
    for (int i=0; i<work.nExcitations; i++) {
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      double tia = work.HijElement[i];
    
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

      double ovlpRatio = getOverlapFactor(I, J, A, B, walk, overlaps, ovlp, false);
      //double ovlpRatio = getOverlapFactor(I, J, A, B, walk, dbig, dbigcopy, false);

      ham += tia * ovlpRatio;
      if (schd.debug) cout << ex1 << "  " << ex2 << "  tia  " << tia << "  ovlpRatio  " << ovlpRatio << endl;

      work.ovlpRatio[i] = ovlpRatio;
    }
    if (schd.debug) cout << endl;
  }

};


#endif
