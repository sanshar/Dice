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
#ifndef PermutedTRWavefunction_HEADER_H
#define PermutedTRWavefunction_HEADER_H
#include "TRWavefunction.h"
#include "PermutedTRWalker.h"

/**
 */
struct PermutedTRWavefunction {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & wave
      & permutations
      & numP
      & characters;
  }

 public:
  
  TRWavefunction wave; 
  MatrixXd permutations; // rows contain permutations except identity
  int numP; // number of permutations not including identity
  VectorXd characters; 

  // default constructor
  PermutedTRWavefunction() {
    wave = TRWavefunction();
    numP = schd.numPermutations;
    characters = VectorXd::Zero(numP);
    int norbs = Determinant::norbs;
    permutations = MatrixXd::Zero(numP, norbs);
    ifstream dump("permutations.txt");
    if (dump) {
     for (int i = 0; i < numP; i++) {
       dump >> characters(i);
       for (int j = 0; j < norbs; j++) dump >> permutations(i, j);
     }
    }
    else {
      if (commrank == 0) cout << "permutations.txt not found!\n";
      exit(0);
    }
  };
  
  Slater& getRef() { return wave.getRef(); }
  Jastrow& getCorr() { return wave.getCorr(); }

  // used at the start of each sampling run
  void initWalker(PermutedTRWalker &walk)  
  {
    walk = PermutedTRWalker(getCorr(), getRef(), permutations);
  }
  
  // used in deterministic calculations
  void initWalker(PermutedTRWalker &walk, Determinant &d) 
  {
    walk = PermutedTRWalker(getCorr(), getRef(), d, permutations);
  }
  
  // used in rdm calculations
  double Overlap(const PermutedTRWalker &walk) const 
  {
    double overlap = wave.Overlap(walk.walkerVec[0]);
    for (int i = 0; i < numP; i++) {
      overlap += characters(i) * wave.Overlap(walk.walkerVec[i+1]);
    }
    return overlap;
  }
 
  // used in HamAndOvlp below
  double Overlap(const PermutedTRWalker &walk, vector<double> &overlaps) const 
  {
    overlaps.resize(numP+1, 0.);
    double totalOverlap = 0.;
    overlaps[0] = wave.Overlap(walk.walkerVec[0]);
    totalOverlap += overlaps[0];
    for (int i = 0; i < numP; i++) {
      overlaps[i+1] = characters(i) * wave.Overlap(walk.walkerVec[i+1]);
      totalOverlap += overlaps[i+1];
    }
    return totalOverlap;
  }

  // used in rdm calculations
  double getOverlapFactor(int i, int a, const PermutedTRWalker& walk, bool doparity) const  
  {
    vector<double> overlaps;
    double totalOverlap = Overlap(walk, overlaps);
    double numerator = wave.getOverlapFactor(i, a, walk.walkerVec[0], doparity) * overlaps[0];
    int norbs = Determinant::norbs;
    for (int n = 0; n < numP; n++) {
      int ip = 2 * permutations(n, i/2) + i%2;
      int ap = 2 * permutations(n, a/2) + a%2;
      numerator += wave.getOverlapFactor(ip, ap, walk.walkerVec[n+1], doparity) * overlaps[n+1];
    }
    return numerator / totalOverlap;
  }

  // used in rdm calculations
  double getOverlapFactor(int I, int J, int A, int B, const PermutedTRWalker& walk, bool doparity) const  
  {
    if (J == 0 && B == 0) return getOverlapFactor(I, A, walk, doparity);
    vector<double> overlaps;
    double totalOverlap = Overlap(walk, overlaps);
    double numerator = wave.getOverlapFactor(I, J, A, B, walk.walkerVec[0], doparity) * overlaps[0];
    int norbs = Determinant::norbs;
    for (int n = 0; n < numP; n++) {
      int ip = 2 * permutations(n, I/2) + I%2;
      int ap = 2 * permutations(n, A/2) + A%2;
      int jp = 2 * permutations(n, J/2) + J%2;
      int bp = 2 * permutations(n, B/2) + B%2;
      numerator += wave.getOverlapFactor(ip, jp, ap, bp, walk.walkerVec[n+1], doparity) * overlaps[n+1];
    }
    return numerator / totalOverlap;
  }
  
  double getOverlapFactor(int i, int a, const PermutedTRWalker& walk, vector<double>& overlaps, double& totalOverlap, bool doparity) const  
  {
    double numerator = wave.getOverlapFactor(i, a, walk.walkerVec[0], doparity) * overlaps[0];
    int norbs = Determinant::norbs;
    for (int n = 0; n < numP; n++) {
      int ip = 2 * permutations(n, i/2) + i%2;
      int ap = 2 * permutations(n, a/2) + a%2;
      numerator += wave.getOverlapFactor(ip, ap, walk.walkerVec[n+1], doparity) * overlaps[n+1];
    }
    return numerator / totalOverlap;
  }

  // used in HamAndOvlp below
  double getOverlapFactor(int I, int J, int A, int B, const PermutedTRWalker& walk, vector<double>& overlaps, double& totalOverlap, bool doparity) const  
  {
    if (J == 0 && B == 0) return getOverlapFactor(I, A, walk, overlaps, totalOverlap, doparity);
    double numerator = wave.getOverlapFactor(I, J, A, B, walk.walkerVec[0], doparity) * overlaps[0];
    int norbs = Determinant::norbs;
    for (int n = 0; n < numP; n++) {
      int ip = 2 * permutations(n, I/2) + I%2;
      int ap = 2 * permutations(n, A/2) + A%2;
      int jp = 2 * permutations(n, J/2) + J%2;
      int bp = 2 * permutations(n, B/2) + B%2;
      numerator += wave.getOverlapFactor(ip, jp, ap, bp, walk.walkerVec[n+1], doparity) * overlaps[n+1];
    }
    return numerator / totalOverlap;
  }
  
  // gradient overlap ratio, used during sampling
  // just calls OverlapWithGradient on the last wave function
  void OverlapWithGradient(const PermutedTRWalker &walk,
                           double &factor,
                           Eigen::VectorXd &grad) const
  {
    vector<double> overlaps;
    double totalOverlap = Overlap(walk, overlaps);
    VectorXd grad_i = 0. * grad;
    wave.OverlapWithGradient(walk.walkerVec[0], factor, grad_i);
    grad += (grad_i * overlaps[0]) / totalOverlap;
    for (int i = 0; i < numP; i++) {
      grad_i = 0. * grad;
      wave.OverlapWithGradient(walk.walkerVec[i+1], factor, grad_i);
      grad += (grad_i * overlaps[i+1]) / totalOverlap;
    }
  }

  void printVariables() const
  {
    wave.printVariables();
  }

  // update after vmc optimization
  // updates the wave function helpers as well
  void updateVariables(Eigen::VectorXd &v) 
  {
    wave.updateVariables(v);
  }

  void getVariables(Eigen::VectorXd &v) const
  {
    wave.getVariables(v);
  }

  long getNumVariables() const
  {
    return wave.getNumVariables();
  }

  string getfileName() const {
    return "PermutedTRWavefunction";
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
  void HamAndOvlp(const PermutedTRWalker &walk,
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
