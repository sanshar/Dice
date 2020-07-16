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
#include <unsupported/Eigen/CXX11/Tensor>
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
  //MatrixXcd intermediate, s;
  double intermediateBuildTime, ciIterationTime;

  JastrowMultiSlater() {
    //intermediate = MatrixXcd::Zero(Determinant::nalpha + Determinant::nbeta, 2*Determinant::norbs);
    //s = MatrixXcd::Zero(Determinant::nalpha + Determinant::nbeta, 2*Determinant::norbs);
    //int nelec = Determinant::nalpha + Determinant::nbeta;
    //int nholes = 2*Determinat::norbs - nelec;
    //intermediate = MatrixXcd::Zero(nelec, 2*Determinant::norbs);
    //s = MatrixXcd::Zero(Determinant::nalpha + Determinant::nbeta, 2*Determinant::norbs);
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
    int nelec = Determinant::nalpha + Determinant::nbeta;
    int nholes = 2*norbs - nelec;

    double refOverlap = walk.walker.refHelper.refOverlap.real(); // < n | phi_0 > 
    ovlp = corr.Overlap(walk.d) * refOverlap;  // < n | psi_0 > 
    double ratio = walk.walker.refHelper.totalOverlap / refOverlap;  // < n | psi > / < n | psi_0 >
    ham = walk.d.Energy(I1, I2, coreE) * ratio;
    double ham1 = 0.; // < n | H' | psi_0 > / < n | psi_0 > , where H' indicates the one particle part
    complex<double> complexHam1(0., 0.); // above ratio without complex projection
    double ham2 = 0.; // < n | H''| psi_0 > / < n | psi_0 > , where H'' indicates the two particle part
    complex<double> complexHam2(0., 0.); // above ratio without complex projection


    //Tensor<complex<double>, 4, RowMajor> h(nelec, nelec, 2*norbs, 2*norbs);
    Tensor<complex<double>, 4, RowMajor> h(nelec, nelec, nholes, nholes);
    h.setZero();
    MatrixXcd intermediate = MatrixXcd::Zero(nelec, nholes);
    intermediate.setZero();
    work.setCounterToZero();
    work.locNorm = ratio;  // < psi | psi > / < psi_0 | psi_0 > sample sqrt
    generateAllScreenedSingleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false);
    generateAllScreenedDoubleExcitation(walk.d, schd.epsilon, schd.screen,
                                        work, false); 
  
    //MatrixXcd intermediate = MatrixXcd::Zero(Determinant::nalpha + Determinant::nbeta, 2*norbs);
    //loop over all the screened excitations
    //if (schd.debug) cout << "eloc excitations\nphi0  d.energy " << ham << endl;
    double initTime = getTime();
    for (int i=0; i<work.nExcitations; i++) {
      int ex1 = work.excitation1[i], ex2 = work.excitation2[i];
      double tia = work.HijElement[i];
    
      int I = ex1 / 2 / norbs, A = ex1 - 2 * norbs * I;
      int J = ex2 / 2 / norbs, B = ex2 - 2 * norbs * J;

      double jia = 0.;
      complex<double> complexRefOverlapRatio(0., 0.);
      double ovlpRatio = 0.;
      if (ex2 == 0) {
        int tableIndexi, tableIndexa;
        walk.walker.refHelper.getRelIndices(I/2, tableIndexi, A/2, tableIndexa, I%2);
        jia =  walk.walker.corrHelper.OverlapRatio(I, A, corr, walk.d, walk.d);
        complexRefOverlapRatio = walk.getDetFactorComplex(I, A, ref); // ratio without complex projection
        intermediate.row(tableIndexi) += tia * jia * walk.walker.refHelper.rtc_b.row(tableIndexa);
        ovlpRatio = jia * (complexRefOverlapRatio * walk.walker.refHelper.refOverlap).real() / refOverlap;  // < m | psi_0 > / < n | psi_0 >
        ham1 += tia * ovlpRatio;
        complexHam1 += tia * jia * complexRefOverlapRatio;
      }
      else {
        int tableIndexi, tableIndexa, tableIndexj, tableIndexb;
        if (I%2 == J%2) {
          int ind00 = min(I/2, J/2), ind01 = min(A/2, B/2);
          int ind10 = max(I/2, J/2), ind11 = max(A/2, B/2);
          walk.walker.refHelper.getRelIndices(ind00, tableIndexi, ind01, tableIndexa, I%2);
          walk.walker.refHelper.getRelIndices(ind10, tableIndexj, ind11, tableIndexb, J%2);
        }
        else if (I%2 == 0) {
          walk.walker.refHelper.getRelIndices(I/2, tableIndexi, A/2, tableIndexa, I%2);
          walk.walker.refHelper.getRelIndices(J/2, tableIndexj, B/2, tableIndexb, J%2);
        }
        else {
          walk.walker.refHelper.getRelIndices(J/2, tableIndexi, B/2, tableIndexa, J%2);
          walk.walker.refHelper.getRelIndices(I/2, tableIndexj, A/2, tableIndexb, I%2);
        }
        jia = walk.walker.corrHelper.OverlapRatio(I, J, A, B, corr, walk.d, walk.d);
        complexRefOverlapRatio = walk.getDetFactorComplex(I, J, A, B, ref); // ratio without complex projection
        ovlpRatio = jia * (complexRefOverlapRatio * walk.walker.refHelper.refOverlap).real() / refOverlap;  // < m | psi_0 > / < n | psi_0 >
        h(tableIndexi, tableIndexj, tableIndexa, tableIndexb) = tia * jia; 
        ham2 += tia * ovlpRatio;
        complexHam2 += tia * jia * complexRefOverlapRatio;
      }
      //double ovlpRatio = jia * walk.getDetFactor(I, A, ref);  // < m | psi_0 > / < n | psi_0 >
      
      //double ovlpRatio = getOverlapFactor(I, J, A, B, walk, false);
      //double ovlpRatio = getOverlapFactor(I, J, A, B, walk, dbig, dbigcopy, false);

      if (schd.debug) cout << I << "  " << A << "  " << J << "  " << B << "  tia  " << tia << "  jia  " << jia << "  ovlpRatio  " << ovlpRatio << endl;

      work.ovlpRatio[i] = ovlpRatio;
    }
    MatrixXcd s = walk.walker.refHelper.t * intermediate;
 
    // double excitation intermediates
    Tensor<complex<double>, 2, RowMajor> t (nelec, nelec);
    Tensor<complex<double>, 2, RowMajor> rt (nholes, nelec);
    Tensor<complex<double>, 2, RowMajor> rtc_b (nholes, nholes);
    //Tensor<complex<double>, 2, RowMajor> rt (2*norbs, nelec);
    //Tensor<complex<double>, 2, RowMajor> rtc_b (2*norbs, 2*norbs);
    
    for (int i = 0; i < nelec; i++)
      for  (int j = 0; j < nelec; j++)
        t(i, j) = walk.walker.refHelper.t(i, j);
    
    for (int i = 0; i < nholes; i++)
      for  (int j = 0; j < nelec; j++)
        rt(i, j) = walk.walker.refHelper.rt(i, j);
    
    for (int i = 0; i < nholes; i++)
      for  (int j = 0; j < nholes; j++) 
        rtc_b(i, j) = walk.walker.refHelper.rtc_b(i, j);
    
    //for (int i = 0; i < 2*norbs; i++)
    //  for  (int j = 0; j < nelec; j++)
    //    rt(i, j) = walk.walker.refHelper.rt(i, j);
    //
    //for (int i = 0; i < 2*norbs; i++)
    //  for  (int j = 0; j < 2*norbs; j++) 
    //    rtc_b(i, j) = walk.walker.refHelper.rtc_b(i, j);
   
    
    Eigen::array<IndexPair<int>, 2> prodDims0 = { IndexPair<int>(2, 0), IndexPair<int>(0, 1) };
    Eigen::array<IndexPair<int>, 2> prodDims1 = { IndexPair<int>(1, 1), IndexPair<int>(2, 0) };
    Tensor<complex<double>, 2, RowMajor> inter0 = h.contract(rt, prodDims0);  
    Tensor<complex<double>, 2, RowMajor> inter1 = h.contract(rt, prodDims1);  
    Eigen::array<IndexPair<int>, 1> matProd = { IndexPair<int>(1, 0) };
    Tensor<complex<double>, 2, RowMajor> d0 = t.contract(inter0, matProd).contract(rtc_b, matProd) - t.contract(inter1, matProd).contract(rtc_b, matProd);
    prodDims0 = { IndexPair<int>(0, 1), IndexPair<int>(3, 0) };
    prodDims1 = { IndexPair<int>(1, 1), IndexPair<int>(3, 0) };
    inter0 = h.contract(rt, prodDims0);  
    inter1 = h.contract(rt, prodDims1);  
    Tensor<complex<double>, 2, RowMajor> d1 = t.contract(inter0, matProd).contract(rtc_b, matProd) - t.contract(inter1, matProd).contract(rtc_b, matProd);
    

    Eigen::array<IndexPair<int>, 1> trans0 = { IndexPair<int>(0, 1) };
    Eigen::array<IndexPair<int>, 1> trans1 = { IndexPair<int>(0, 0) };
    Tensor<complex<double>, 4, RowMajor> hbar = h.contract(t, trans0).contract(t, trans0).contract(rtc_b, trans1).contract(rtc_b, trans1);
    //Tensor<complex<double>, 4, RowMajor> d2 (nelec, nelec, 2*norbs, 2*norbs);
    Tensor<complex<double>, 4, RowMajor> d2 (nelec, nelec, nholes, nholes);
    for (int i = 0; i < nelec; i++) {
      for (int j = 0; j < nelec; j++) {
        for (int a  = 0; a < nholes; a++) { 
          for (int b = 0; b < nholes; b++) {
            d2(i,j,a,b) = hbar(i,j,a,b) - hbar(i,j,b,a) - hbar(j,i,a,b) + hbar(j,i,b,a); 
          }
        }
      }
    }
    //for (int i = 0; i < nelec; i++) {
    //  for (int j = 0; j < nelec; j++) {
    //    for (int a  = 0; a < 2*norbs; a++) { 
    //      for (int b = 0; b < 2*norbs; b++) {
    //        d2(i,j,a,b) = hbar(i,j,a,b) - hbar(i,j,b,a) - hbar(j,i,a,b) + hbar(j,i,b,a); 
    //      }
    //    }
    //  }
    //}
    //

    if (schd.debug) {
      cout << "s\n" << s << endl << endl;
      cout << "d0\n" << d0 << endl << endl;
      cout << "d1\n" << d1 << endl << endl;
      cout << "h\n\n";
      for (int i = 0; i < nelec; i++) {
        for (int  j = 0; j < nelec; j++) {
          cout << "slice " << i << "  " << j << endl;
          for (int a = 0; a < nholes; a++) {
            for (int b = 0; b < nholes; b++) {
              cout << h(i, j, a, b) << " ";
            }
            cout << endl;
          }
          cout << endl;
        }
      }

      cout << "d2\n\n";
      for (int i = 0; i < nelec; i++) {
        for (int  j = 0; j < nelec; j++) {
          cout << "slice " << i << "  " << j << endl;
          for (int a = 0; a < nholes; a++) {
            for (int b = 0; b < nholes; b++) {
              cout << d2(i, j, a, b) << " ";
            }
            cout << endl;
          }
          cout << endl;
        }
      }
    }
    intermediateBuildTime += (getTime() - initTime);
   

    initTime = getTime();
    double locES1 = ref.ciCoeffs[0] * ham1; // singles local energy
    double locES2 = ref.ciCoeffs[0] * ham2; // doubles local energy
    size_t count4 = 0;
    for (int i = 1; i < ref.numDets; i++) {
      int rank = ref.ciExcitations[i][0].size();
      complex<double> laplaceDet1(0., 0.);
      complex<double> laplaceDet20(0., 0.);
      complex<double> laplaceDet21(0., 0.);
      complex<double> laplaceDet22(0., 0.);
      if (rank == 1) { 
        laplaceDet1 = -s(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]);
        laplaceDet20 = -d0(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]);
        laplaceDet21 = d1(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]);
      }
      else if (rank == 2) { 
        laplaceDet1 = -(s(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]) * walk.walker.refHelper.tc(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][1])
                     - s(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][1]) * walk.walker.refHelper.tc(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][0])
                     + walk.walker.refHelper.tc(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]) * s(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][1])
                     - walk.walker.refHelper.tc(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][1]) * s(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][0]));
        laplaceDet20 = -(d0(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]) * walk.walker.refHelper.tc(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][1])
                     - d0(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][1]) * walk.walker.refHelper.tc(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][0])
                     + walk.walker.refHelper.tc(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]) * d0(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][1])
                     - walk.walker.refHelper.tc(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][1]) * d0(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][0]));
        laplaceDet21 = d1(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]) * walk.walker.refHelper.tc(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][1])
                     - d1(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][1]) * walk.walker.refHelper.tc(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][0])
                     + walk.walker.refHelper.tc(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][0]) * d1(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][1])
                     - walk.walker.refHelper.tc(ref.ciExcitations[i][0][0], ref.ciExcitations[i][1][1]) * d1(ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][0]);
        laplaceDet22 = d2(ref.ciExcitations[i][0][0], ref.ciExcitations[i][0][1], ref.ciExcitations[i][1][0], ref.ciExcitations[i][1][1]);
      }
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
          laplaceDet1 -= temp.determinant();
          
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = d0(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet20 -= temp.determinant();
          
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = d1(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet21 += temp.determinant();
        }

        for (int mu = 0; mu < rank; mu++) {
          for (int nu = mu + 1; nu < rank; nu++) {
            double parity_munu = ((mu + nu + 1)%2 == 0) ? 1. : -1.;
            for (int t = 0; t < rank; t++) {
              for (int u = t + 1; u < rank; u++) {
                double parity_tu = ((t + u + 1)%2 == 0) ? 1. : -1.;
                laplaceDet22 += parity_munu * parity_tu * d2(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][0][nu], ref.ciExcitations[i][1][t], ref.ciExcitations[i][1][u])
                              * tcSlice(3 - mu - nu, 3 - t - u);
              }
            }
          }
        }

      }
      else if (rank == 4) {
        Matrix4cd temp;
        for (int mu = 0; mu < rank; mu++) {
          temp = walk.walker.refHelper.tcSlice[count4];
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = s(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet1 -= temp.determinant();
          
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = d0(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet20 -= temp.determinant();
          
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = d1(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet21 += temp.determinant();
        }
        
        for (int mu = 0; mu < rank; mu++) {
          for (int nu = mu + 1; nu < rank; nu++) {
            double parity_munu = ((mu + nu + 1)%2 == 0) ? 1. : -1.;
            int mup, nup;
            if (mu == 0) {
              mup = 3 - nu + nu/3; nup = 6 - nu - mup;
            }
            else {
              mup = 0; nup = 6 - mu - nu;
            }
            for (int t = 0; t < rank; t++) {
              for (int u = t + 1; u < rank; u++) {
                double parity_tu = ((t + u + 1)%2 == 0) ? 1. : -1.;
                int tp, up;
                if (t == 0) {
                  tp = 3 - u + u/3; up = 6 - u - tp;
                }
                else {
                  tp = 0; up = 6 - t - u;
                }
                laplaceDet22 += parity_munu * parity_tu * d2(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][0][nu], ref.ciExcitations[i][1][t], ref.ciExcitations[i][1][u])
                              * ( walk.walker.refHelper.tcSlice[count4](mup, tp) * walk.walker.refHelper.tcSlice[count4](nup, up) 
                                - walk.walker.refHelper.tcSlice[count4](mup, up) * walk.walker.refHelper.tcSlice[count4](nup, tp));
              }
            }
          }
        }
        
        count4++;
      }
      else {
        MatrixXcd tcSlice = MatrixXcd::Zero(rank, rank);
        //igl::slice(walk.walker.refHelper.tc, ref.ciExcitations[i][0], ref.ciExcitations[i][1], tcSlice);
        for (int mu = 0; mu < rank; mu++) 
          for (int nu = 0; nu < rank; nu++) 
            tcSlice(mu, nu) = walk.walker.refHelper.tc(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][nu]);
        
        MatrixXcd temp = MatrixXcd::Zero(rank, rank);
        for (int mu = 0; mu < rank; mu++) {
          temp = tcSlice;
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = s(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet1 -= temp.determinant();
          
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = d0(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet20 -= temp.determinant();
          
          for (int t = 0; t < rank; t++) {
            temp(mu, t) = d1(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][1][t]);
          }
          laplaceDet21 += temp.determinant();
          
          vector<int> range(rank);
          for (int mu = 0; mu < rank; mu++) range[mu] = mu;
          for (int mu = 0; mu < rank; mu++) {
            for (int nu = mu + 1; nu < rank; nu++) {
              double parity_munu = ((mu + nu + 1)%2 == 0) ? 1. : -1.;
              vector<int> munu = {mu, nu};
              vector<int> diff0;
              std::set_difference(range.begin(), range.end(), munu.begin(), munu.end(), std::inserter(diff0, diff0.begin()));
              for (int t = 0; t < rank; t++) {
                for (int u = t + 1; u < rank; u++) {
                  double parity_tu = ((t + u + 1)%2 == 0) ? 1. : -1.;
                  vector<int> tu = {t, u};
                  vector<int> diff1;
                  std::set_difference(range.begin(), range.end(), tu.begin(), tu.end(), std::inserter(diff1, diff1.begin()));
                  
                  MatrixXcd tcSliceSlice = MatrixXcd::Zero(rank - 2, rank - 2);
                  //igl::slice(walk.walker.refHelper.tc, ref.ciExcitations[i][0], ref.ciExcitations[i][1], tcSlice);
                  for (int mup = 0; mup < rank - 2; mup++) 
                    for (int nup = 0; nup < rank - 2; nup++) 
                      tcSliceSlice(mup, nup) = tcSlice(diff0[mup], diff1[nup]);
                  
                  laplaceDet22 += parity_munu * parity_tu * d2(ref.ciExcitations[i][0][mu], ref.ciExcitations[i][0][nu], ref.ciExcitations[i][1][t], ref.ciExcitations[i][1][u])
                                * tcSliceSlice.determinant();
                }
              }
            }
          }
        }
      }
      locES1 += ref.ciCoeffs[i] * ((walk.walker.refHelper.ciOverlapRatios[i] * complexHam1 + ref.ciParity[i] * laplaceDet1) * walk.walker.refHelper.refOverlap).real() / refOverlap;
      locES2 += ref.ciCoeffs[i] * ((walk.walker.refHelper.ciOverlapRatios[i] * complexHam2 + ref.ciParity[i] * (laplaceDet20 + laplaceDet21 + laplaceDet22)) * walk.walker.refHelper.refOverlap).real() / refOverlap;
    }
    ciIterationTime += (getTime() - initTime);
    ham += (locES1 + locES2);
    ham *= ratio;
  }

  
};


#endif
