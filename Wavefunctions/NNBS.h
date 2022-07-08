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
#ifndef NNBS_HEADER_H
#define NNBS_HEADER_H
#include "NNBSWalker.h"
#include <algorithm>

using namespace std;
using namespace std::placeholders;
using namespace Eigen;

// determinant cost function
double costDet(VectorXd& v, const MatrixXd& slice) {
  int norbs = Determinant::norbs;
  int nelec = Determinant::nalpha + Determinant::nbeta;
  MatrixXd bfSlice = Map<MatrixXd>(v.data(), nelec, nelec) + slice;
  FullPivLU<MatrixXd> lua(bfSlice);
  return lua.determinant();
}

// this returns gradient ratio
void costDetGradient(VectorXd& v, VectorXd& grad, const MatrixXd& slice) {
  int norbs = Determinant::norbs;
  int nelec = Determinant::nalpha + Determinant::nbeta;
  MatrixXd bfSlice = Map<MatrixXd>(v.data(), nelec, nelec) + slice;
  FullPivLU<MatrixXd> lua(bfSlice);
  MatrixXd deriv = lua.inverse().transpose();
  grad = Map<VectorXd>(deriv.data(), nelec*nelec);
}

/**
 */
struct NNBS {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & fnnb
      & moCoeffs;
  }

 public:
  fnn fnnb;
  MatrixXd moCoeffs;

  // default constructor
  NNBS() {
    int norbs = Determinant::norbs;
    int nelec = Determinant::nalpha + Determinant::nbeta;
    vector<int> sizes = {2*norbs};    
    for (int i = 0; i < schd.numHiddenLayers; i++) 
      sizes.push_back(schd.numHidden);
    sizes.push_back(nelec * nelec);
    fnnb = fnn(sizes);
    int size = 2*norbs;
    moCoeffs = MatrixXd::Zero(size, size);
    readMat(moCoeffs, "hf.txt");
  };
  
  // no ref and corr for this wave function
  MatrixXd& getRef() { return moCoeffs; }
  fnn& getCorr() { return fnnb; }

  // used at the start of each sampling run
  void initWalker(NNBSWalker &walk)  
  {
    walk = NNBSWalker(moCoeffs);
  }
  
  // used in deterministic calculations
  void initWalker(NNBSWalker &walk, Determinant &d) 
  {
    walk = NNBSWalker(moCoeffs, d);
  }
  
  // used in HamAndOvlp below
  // used in rdm calculations
  double Overlap(const NNBSWalker &walk) const 
  {
    return fnnb.evaluate(walk.occ, std::bind(costDet, std::placeholders::_1, walk.occSlice));
  }
 
  // used in rdm calculations
  double getOverlapFactor(int i, int a, const NNBSWalker& walk, bool doparity) const  
  {
    return 0.;
  }

  // used in rdm calculations
  double getOverlapFactor(int I, int J, int A, int B, const NNBSWalker& walk, bool doparity) const  
  {
    return 0.;
  }
  
  
  // gradient overlap ratio, used during sampling
  void OverlapWithGradient(const NNBSWalker &walk,
                           double &factor,
                           Eigen::VectorXd &grad) const
  {
    fnnb.backPropagate(walk.occ, std::bind(costDetGradient, std::placeholders::_1, std::placeholders::_2, walk.occSlice), grad);
  }

  void printVariables() const
  {
    cout << fnnb;
  }

  // update after vmc optimization
  void updateVariables(Eigen::VectorXd &v) 
  {
    fnnb.updateVariables(v);
  }

  void getVariables(Eigen::VectorXd &v) const
  {
    fnnb.getVariables(v);
  }

  long getNumVariables() const
  {
    return fnnb.getNumVariables();
  }

  string getfileName() const {
    return "NNBS";
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
  void HamAndOvlp(const NNBSWalker &walk,
                  double &ovlp, double &ham, 
                  workingArray& work, bool fillExcitations=true, double epsilon = schd.epsilon) const
  {
    int norbs = Determinant::norbs;

    ovlp = Overlap(walk);
    ham = walk.det.Energy(I1, I2, coreE); 

    work.setCounterToZero();
    generateAllScreenedSingleExcitation(walk.det, epsilon, schd.screen,
                                        work, true);
    generateAllScreenedDoubleExcitation(walk.det, epsilon, schd.screen,
                                        work, true);
  
    //loop over all the screened excitations
    if (schd.debug) {
      cout << "eloc excitations" << endl;
      cout << "phi0  d.energy " << ham << endl;
    }

    for (int i=0; i<work.nExcitations; i++) {
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      double tia = work.HijElement[i];
    
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;
      
      NNBSWalker excitedWalker = walk;
      excitedWalker.updateWalker(moCoeffs, fnnb, ex1, ex2);

      double ovlpRatio = Overlap(excitedWalker) / ovlp;

      ham += tia * ovlpRatio;
      if (schd.debug) cout << ex1 << "  " << ex2 << "  tia  " << tia << "  ovlpRatio  " << ovlpRatio << endl;

      work.ovlpRatio[i] = ovlpRatio;
    }
    if (schd.debug) cout << endl;
  }
  
  void HamAndOvlpLanczos(const NNBSWalker &walk,
                         Eigen::VectorXd &lanczosCoeffsSample,
                         double &ovlpSample,
                         workingArray& work,
                         workingArray& moreWork, double &alpha)
  {
    return ;
  }
};


#endif
