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
#ifndef JastrowMultiSlater_HEADER_H
#define JastrowMultiSlater_HEADER_H
#include <vector>
#include <set>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <unordered_set>
#include "igl/slice.h"
#include "JastrowMultiSlaterWalker.h"

class oneInt;
class twoInt;
class twoIntHeatBathSHM;
class workingArray;
class Determinant;


struct JastrowMultiSlater {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & corr
        & ref;
  }

 public:
  
  Jastrow corr; //The jastrow factors
  MultiSlater ref; //reference
  MatrixXcd intermediate, s;
  double intermediateBuildTime, ciIterationTime;

  JastrowMultiSlater() {
    intermediate = MatrixXcd::Zero(Determinant::nalpha + Determinant::nbeta, 2*Determinant::norbs);
    s = MatrixXcd::Zero(Determinant::nalpha + Determinant::nbeta, 2*Determinant::norbs);
    intermediateBuildTime = 0.;
    ciIterationTime = 0.;
  };
  MultiSlater& getRef() { return ref; }
  Jastrow& getCorr() { return corr; }

  void initWalker(JastrowMultiSlaterWalker &walk)  
  {
    walk = JastrowMultiSlaterWalker(corr, ref);
  }
  
  void initWalker(JastrowMultiSlaterWalker &walk, Determinant &d) 
  {
    walk = JastrowMultiSlaterWalker(corr, ref, d);
  }
  
  /**
   *This calculates the overlap of the walker with the
   *jastrow and the ciexpansion 
   */
  double Overlap(const JastrowMultiSlaterWalker &walk) const 
  {
    return corr.Overlap(walk.d) * walk.getDetOverlap(ref);
  }

  // these det ratio functions won't work for rdm calculations, they are only meant to be used in hamandoverlap below
  // used in hamandvolp below
  // returns < m | psi_0 > / < n | psi_0 > with complex projection
  double getOverlapFactor(int i, int a, const JastrowMultiSlaterWalker& walk, bool doparity) const  
  {
    return walk.walker.corrHelper.OverlapRatio(i, a, corr, walk.d, walk.d)
        * walk.getDetFactor(i, a, ref);
  }

  double getOverlapFactor(int I, int J, int A, int B, const JastrowMultiSlaterWalker& walk, bool doparity) const  
  {
    return 0.;
  }
  
  
  double getOverlapFactor(const JastrowMultiSlaterWalker& walk, std::array<unordered_set<int>, 2> &from, std::array<unordered_set<int>, 2> &to) const
  {
    //cout << "\ndet  " << walk.getDetFactor(from, to) << "  corr  " << walk.corrHelper.OverlapRatio(from, to, corr) << endl;
    return 0.;
  }

  /**
   * This basically calls the overlapwithgradient(determinant, factor, grad)
   */
  void OverlapWithGradient(const JastrowMultiSlaterWalker &walk,
                           double &factor,
                           Eigen::VectorXd &grad) const
  {
    double factor1 = 1.0;
    Eigen::VectorBlock<VectorXd> gradhead = grad.head(corr.getNumVariables());
    corr.OverlapWithGradient(walk.d, gradhead, factor1);
    gradhead *= walk.walker.refHelper.totalOverlap / walk.walker.refHelper.refOverlap.real();

    Eigen::VectorBlock<VectorXd> gradtail = grad.tail(grad.rows() - corr.getNumVariables());
    walk.OverlapWithGradient(ref, gradtail);
  }

  void printVariables() const
  {
    corr.printVariables();
    ref.printVariables();
  }

  void updateVariables(Eigen::VectorXd &v) 
  {
    Eigen::VectorBlock<VectorXd> vhead = v.head(corr.getNumVariables());
    corr.updateVariables(vhead);
    Eigen::VectorBlock<VectorXd> vtail = v.tail(v.rows() - corr.getNumVariables());
    ref.updateVariables(vtail);
  }

  void getVariables(Eigen::VectorXd &v) const
  {
    if (v.rows() != getNumVariables())
      v = VectorXd::Zero(getNumVariables());

    Eigen::VectorBlock<VectorXd> vhead = v.head(corr.getNumVariables());
    corr.getVariables(vhead);
    Eigen::VectorBlock<VectorXd> vtail = v.tail(v.rows() - corr.getNumVariables());
    ref.getVariables(vtail);
  }


  long getNumJastrowVariables() const
  {
    return corr.getNumVariables();
  }
  //factor = <psi|w> * prefactor;

  long getNumVariables() const
  {
    int norbs = Determinant::norbs;
    long numVars = 0;
    numVars += getNumJastrowVariables();
    numVars += ref.getNumVariables();

    return numVars;
  }

  string getfileName() const {
    return ref.getfileName() + corr.getfileName();
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


  // this needs to be cleaned up, some of the necessary functions already exist
  // only single excitations for now
  void HamAndOvlp(const JastrowMultiSlaterWalker &walk,
                  double &ovlp, double &ham, 
                  workingArray& work, bool fillExcitations=true) 
  {
    int norbs = Determinant::norbs;

    double refOverlap = walk.walker.refHelper.refOverlap.real(); // < n | phi_0 > 
    ovlp = corr.Overlap(walk.d) * refOverlap;  // < n | psi_0 > 
    double ratio = walk.walker.refHelper.totalOverlap / refOverlap;  // < n | psi > / < n | psi_0 >
    ham = walk.d.Energy(I1, I2, coreE) * ratio;
    double ham0 = 0.; // < n | H' | psi_0 > / < n | psi_0 > , where H' indicates H w/o diagonal element

    intermediate.setZero();
    work.setCounterToZero();
    work.locNorm = ratio;  // < psi | psi > / < psi_0 | psi_0 > sample sqrt
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);
    //generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
    //                                    work, false); 
  
    //MatrixXcd intermediate = MatrixXcd::Zero(Determinant::nalpha + Determinant::nbeta, 2*norbs);
    //loop over all the screened excitations
    //if (schd.debug) cout << "eloc excitations\nphi0  d.energy " << ham << endl;
    double initTime = getTime();
    for (int i=0; i<work.nExcitations; i++) {
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      double tia = work.HijElement[i];
    
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int tableIndexi, tableIndexa;
      walk.walker.refHelper.getRelIndices(I/2, tableIndexi, A/2, tableIndexa, I%2);
      //int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

      double jia =  walk.walker.corrHelper.OverlapRatio(I, A, corr, walk.d, walk.d);
      double ovlpRatio = jia * walk.getDetFactor(I, A, ref);  // < m | psi_0 > / < n | psi_0 >
      
      //double ovlpRatio = getOverlapFactor(I, J, A, B, walk, false);
      //double ovlpRatio = getOverlapFactor(I, J, A, B, walk, dbig, dbigcopy, false);

      ham0 += tia * ovlpRatio;
      intermediate.row(tableIndexi) += tia * jia * walk.walker.refHelper.rtc_b.row(tableIndexa);
      if (schd.debug) cout << ex1 << "  " << ex2 << "  tia  " << tia << "  jia  " << jia << "  ovlpRatio  " << ovlpRatio << endl;

      work.ovlpRatio[i] = ovlpRatio;
    }
    s = walk.walker.refHelper.t * intermediate;
    intermediateBuildTime += (getTime() - initTime);
   

    initTime = getTime();
    double locES = ref.ciCoeffs[0] * ham0; // singles local energy
    size_t count4 = 0;
    for (int i = 1; i < ref.numDets; i++) {
      int rank = ref.ciExcitations[i][0].size();
      complex<double> laplaceDet(0., 0.);
      if (rank == 1) 
        laplaceDet = -s(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]);
      else if (rank == 2) 
        laplaceDet = -(s(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]) * walk.walker.refHelper.tc(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][1])
                     - s(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][1]) * walk.walker.refHelper.tc(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][0])
                     + walk.walker.refHelper.tc(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]) * s(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][1])
                     - walk.walker.refHelper.tc(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][1]) * s(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][0]));
      else if (rank == 3) {
        Matrix3cd temp;
        Matrix3cd tcSlice;
        //igl::slice(walk.walker.refHelper.tc, ref.ciExcitations[i][0], ref.ciExcitations[i][1], tcSlice);
        for (int mu = 0; mu < rank; mu++) 
          for (int nu = 0; nu < rank; nu++) 
            tcSlice(mu, nu) = walk.walker.refHelper.tc(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][nu]);
        
        for (int mu = 0; mu < rank; mu++) {
          temp = tcSlice;
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = s(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet -= temp.determinant();
        }
      }
      else if (rank == 4) {
        Matrix4cd temp;
        for (int mu = 0; mu < rank; mu++) {
          temp = walk.walker.refHelper.tcSlice[count4];
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = s(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet -= temp.determinant();
        }
        count4++;
      }
      else {
        MatrixXcd tcSlice;
        igl::slice(walk.walker.refHelper.tc, ref.ciExcitations[i][0], ref.ciExcitations[i][1], tcSlice);
        for (int mu = 0; mu < rank; mu++) {
          auto temp = tcSlice;
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = s(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet -= calcDet(temp);
        }
      }
      locES += ref.ciCoeffs[i] * (walk.walker.refHelper.ciOverlaps[i] * ham0 + ref.ciParity[i] * (laplaceDet * walk.walker.refHelper.refOverlap).real()) / refOverlap;
    }
    ciIterationTime += (getTime() - initTime);
    ham += locES;
    ham *= ratio;
  }

  
};


#endif
