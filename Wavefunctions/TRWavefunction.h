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
#ifndef TRWavefunction_HEADER_H
#define TRWavefunction_HEADER_H
#include "CorrelatedWavefunction.h"
#include "TRWalker.h"

/**
 This is a linear combination of Jastrow Slater correlated wave functions.
 Only the last element of the list is optimized
 */
struct TRWavefunction {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & wave;
  }

 public:
  
  CorrelatedWavefunction<Jastrow, Slater> wave; 
  // default constructor
  TRWavefunction() {
   wave = CorrelatedWavefunction<Jastrow, Slater>();
  };
  
  Slater& getRef() { return wave.ref; }
  Jastrow& getCorr() { return wave.corr; }

  // used at the start of each sampling run
  void initWalker(TRWalker &walk)  
  {
    walk = TRWalker(wave.corr, wave.ref);
  }
  
  // used in deterministic calculations
  void initWalker(TRWalker &walk, Determinant &d) 
  {
    walk = TRWalker(wave.corr, wave.ref, d);
  }
  
  // used in rdm calculations
  double Overlap(const TRWalker &walk) const 
  {
    return wave.Overlap(walk.walkerPair[0]) + wave.Overlap(walk.walkerPair[0]);
  }
 
  // used in HamAndOvlp below
  double Overlap(const TRWalker &walk, array<double, 2> &overlaps) const 
  {
    overlaps[0] = wave.Overlap(walk.walkerPair[0]);
    overlaps[1] = wave.Overlap(walk.walkerPair[1]);
    return overlaps[0] + overlaps[1];
  }

  // used in rdm calculations
  double getOverlapFactor(int i, int a, const TRWalker& walk, bool doparity) const  
  {
    array<double, 2> overlaps;
    double totalOverlap = Overlap(walk, overlaps);
    double numerator = wave.getOverlapFactor(i, a, walk.walkerPair[0], doparity) * overlaps[0];
    int norbs = Determinant::norbs;
    if (i%2 == 0) i += 1;
    else i -= 1;
    if (a%2 == 0) a += 1;
    else a -= 1;
    numerator += wave.getOverlapFactor(i, a, walk.walkerPair[1], doparity) * overlaps[1];
    return numerator / totalOverlap;
  }

  // used in rdm calculations
  double getOverlapFactor(int I, int J, int A, int B, const TRWalker& walk, bool doparity) const  
  {
    if (J == 0 && B == 0) return getOverlapFactor(I, A, walk, doparity);
    array<double, 2> overlaps;
    double totalOverlap = Overlap(walk, overlaps);
    double numerator = wave.getOverlapFactor(I, J, A, B, walk.walkerPair[0], doparity) * overlaps[0];
    int norbs = Determinant::norbs;
    if (I%2 == 0) I += 1;
    else I -= 1;
    if (A%2 == 0) A += 1;
    else A -= 1;
    if (J%2 == 0) J += 1;
    else J -= 1;
    if (B%2 == 0) B += 1;
    else B -= 1;
    numerator += wave.getOverlapFactor(I, J, A, B, walk.walkerPair[1], doparity) * overlaps[1];
    return numerator / totalOverlap;
  }
  
  double getOverlapFactor(int i, int a, const TRWalker& walk, array<double, 2>& overlaps, double& totalOverlap, bool doparity) const  
  {
    double numerator = wave.getOverlapFactor(i, a, walk.walkerPair[0], doparity) * overlaps[0];
    int norbs = Determinant::norbs;
    if (i%2 == 0) i += 1;
    else i -= 1;
    if (a%2 == 0) a += 1;
    else a -= 1;
    numerator += wave.getOverlapFactor(i, a, walk.walkerPair[1], doparity) * overlaps[1];
    return numerator / totalOverlap;
  }

  // used in HamAndOvlp below
  double getOverlapFactor(int I, int J, int A, int B, const TRWalker& walk, array<double, 2>& overlaps, double& totalOverlap, bool doparity) const  
  {
    if (J == 0 && B == 0) return getOverlapFactor(I, A, walk, overlaps, totalOverlap, doparity);
    double numerator = wave.getOverlapFactor(I, J, A, B, walk.walkerPair[0], doparity) * overlaps[0];
    int norbs = Determinant::norbs;
    if (I%2 == 0) I += 1;
    else I -= 1;
    if (A%2 == 0) A += 1;
    else A -= 1;
    if (J%2 == 0) J += 1;
    else J -= 1;
    if (B%2 == 0) B += 1;
    else B -= 1;
    numerator += wave.getOverlapFactor(I, J, A, B, walk.walkerPair[1], doparity) * overlaps[1];
    return numerator / totalOverlap;
  }
  
  // gradient overlap ratio, used during sampling
  // just calls OverlapWithGradient on the last wave function
  void OverlapWithGradient(const TRWalker &walk,
                           double &factor,
                           Eigen::VectorXd &grad) const
  {
    array<double, 2> overlaps;
    double totalOverlap = Overlap(walk, overlaps);
    size_t index = 0;
    VectorXd grad_0 = 0. * grad, grad_1 = 0. * grad;
    wave.OverlapWithGradient(walk.walkerPair[0], factor, grad_0);
    wave.OverlapWithGradient(walk.walkerPair[1], factor, grad_1);
    grad = (grad_0 * overlaps[0] + grad_1 * overlaps[1]) / totalOverlap;
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
    return "TRWavefunction";
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
  void HamAndOvlp(const TRWalker &walk,
                  double &ovlp, double &ham, 
                  workingArray& work, bool fillExcitations=true) const
  {
    int norbs = Determinant::norbs;

    array<double, 2> overlaps;
    ovlp = Overlap(walk, overlaps);
    if (schd.debug) {
      cout << "overlaps\n";
      for (int i = 0; i <overlaps.size(); i++) cout << overlaps[i] << "  ";
      cout << endl;
    }
    ham = walk.walkerPair[0].d.Energy(I1, I2, coreE); 

    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.walkerPair[0].d, schd.epsilon, schd.screen,
                                        work, false);
    generateAllScreenedDoubleExcitation(walk.walkerPair[0].d, schd.epsilon, schd.screen,
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
