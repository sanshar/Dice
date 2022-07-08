/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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
#include "integral.h"
#include "input.h"
#include "string.h"
#include "global.h"
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include "math.h"
#include "boost/format.hpp"
#include <fstream>
#include "Determinants.h"
#include "SHCIshm.h"
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include <boost/serialization/vector.hpp>

using namespace boost;

bool myfn(double i, double j) { return fabs(i)<fabs(j); }



//=============================================================================
void readIntegralsAndInitializeDeterminantStaticVariables(string fcidump) {
//-----------------------------------------------------------------------------
    /*!
    Read FCIDUMP file and populate "I1, I2, coreE, nelec, norbs, irrep"

    :Inputs:

        string fcidump:
            Name of the FCIDUMP file
    */
//-----------------------------------------------------------------------------
  
  if (H5Fis_hdf5(fcidump.c_str())) {
    readIntegralsHDF5AndInitializeDeterminantStaticVariables(fcidump);
    return;
  }

#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  ifstream dump(fcidump.c_str());
  if (!dump.good()) {
    cout << "Integral file "<<fcidump<<" does not exist!"<<endl;
    exit(0);
  }
  vector<int> irrep;
  int nelec, sz, norbs, nalpha, nbeta;

#ifndef SERIAL
  if (commrank == 0) {
#endif
    I2.ksym = false;
    bool startScaling = false;
    norbs = -1;
    nelec = -1;
    sz = -1;

    int index = 0;
    vector<string> tok;
    string msg;
    while(!dump.eof()) {
      std::getline(dump, msg);
      trim(msg);
      boost::split(tok, msg, is_any_of(", \t="), token_compress_on);

      if (startScaling == false && tok.size() == 1 && (boost::iequals(tok[0],"&END") || boost::iequals(tok[0], "/"))) {
        startScaling = true;
        index += 1;
        break;
      } else if(startScaling == false) {
        if (boost::iequals(tok[0].substr(0,4),"&FCI")) {
          if (boost::iequals(tok[1].substr(0,4), "NORB"))
            norbs = atoi(tok[2].c_str());
          if (boost::iequals(tok[3].substr(0,5), "NELEC"))
            nelec = atoi(tok[4].c_str());
          if (boost::iequals(tok[5].substr(0,5), "MS2"))
            sz = atoi(tok[6].c_str());
        } else if (boost::iequals(tok[0].substr(0,4),"ISYM")) {
          continue;
        } else if (boost::iequals(tok[0].substr(0,4),"KSYM")) {
          I2.ksym = true;
        } else if (boost::iequals(tok[0].substr(0,6),"ORBSYM")) {
          for (int i=1;i<tok.size(); i++)
            irrep.push_back(atoi(tok[i].c_str()));
        } else {
          for (int i=0;i<tok.size(); i++)
            irrep.push_back(atoi(tok[i].c_str()));
        }
        index += 1;
      }
    } // while

    if (norbs == -1 || nelec == -1 || sz == -1) {
      std::cout << "could not read the norbs or nelec or MS2"<<std::endl;
      exit(0);
    }
    nalpha = nelec/2 + sz/2;
    nbeta = nelec - nalpha;
    irrep.resize(norbs);
#ifndef SERIAL
  } // commrank=0

  mpi::broadcast(world, nalpha, 0);
  mpi::broadcast(world, nbeta, 0);
  mpi::broadcast(world, nelec, 0);
  mpi::broadcast(world, norbs, 0);
  mpi::broadcast(world, irrep, 0);
  mpi::broadcast(world, I2.ksym, 0);
#endif

  long npair = norbs*(norbs+1)/2;
  if (I2.ksym) npair = norbs*norbs;
  I2.norbs = norbs;
  I2.npair = npair;
  int inner = norbs;
  if (schd.nciAct > 0) inner = schd.nciCore + schd.nciAct;
  I2.inner = inner;
  //innerDetLen = DetLen;
  int virt = norbs - inner;
  I2.virt = virt;
  //size_t nvirtpair = virt*(virt+1)/2;
  size_t nii = inner*(inner+1)/2;
  size_t niv = inner*virt;
  size_t nvv = virt*(virt+1)/2;
  size_t niiii = nii*(nii+1)/2;
  size_t niiiv = nii*niv;
  size_t niviv = niv*(niv+1)/2;
  size_t niivv = nii*nvv;
  I2.nii = nii;
  I2.niv = niv;
  I2.nvv = nvv;
  I2.niiii = niiii;
  I2.niiiv = niiiv;
  I2.niviv = niviv;
  I2.niivv = niivv;
  size_t I2memory = niiii + niiiv + niviv + niivv;
  //size_t I2memory = npair*(npair+1)/2 - nvirtpair*(nvirtpair+1)/2; //memory in bytes

#ifndef SERIAL
  world.barrier();
#endif

  int2Segment.truncate((I2memory)*sizeof(double));
  regionInt2 = boost::interprocess::mapped_region{int2Segment, boost::interprocess::read_write};
  memset(regionInt2.get_address(), 0., (I2memory)*sizeof(double));

#ifndef SERIAL
  world.barrier();
#endif

  I2.store = static_cast<double*>(regionInt2.get_address());

  if (commrank == 0) {

    I1.store.clear();
    I1.store.resize(2*norbs*(2*norbs),0.0); I1.norbs = 2*norbs;
    coreE = 0.0;

    vector<string> tok;
    string msg;
    while(!dump.eof()) {
      std::getline(dump, msg);
      trim(msg);
      boost::split(tok, msg, is_any_of(", \t"), token_compress_on);
      if (tok.size() != 5) continue;

      double integral = atof(tok[0].c_str());int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str()), c=atoi(tok[3].c_str()), d=atoi(tok[4].c_str());

      if(a==b&&b==c&&c==d&&d==0) {
        coreE = integral;
      } else if (b==c&&c==d&&d==0) {
        continue;//orbital energy
      } else if (c==d&&d==0) {
        I1(2*(a-1),2*(b-1)) = integral; //alpha,alpha
        I1(2*(a-1)+1,2*(b-1)+1) = integral; //beta,beta
        I1(2*(b-1),2*(a-1)) = integral; //alpha,alpha
        I1(2*(b-1)+1,2*(a-1)+1) = integral; //beta,beta
      } else {
        int n = 0;
        if ((a-1) >= inner) n++;
        if ((b-1) >= inner) n++;
        if ((c-1) >= inner) n++;
        if ((d-1) >= inner) n++;
        if (n < 3) I2(2*(a-1),2*(b-1),2*(c-1),2*(d-1)) = integral;
      }
    } // while

    //exit(0);
    I2.maxEntry = *std::max_element(&I2.store[0], &I2.store[0]+I2memory,myfn);
    I2.Direct = MatrixXd::Zero(norbs, norbs); I2.Direct *= 0.;
    I2.Exchange = MatrixXd::Zero(norbs, norbs); I2.Exchange *= 0.;

    for (int i=0; i<inner; i++)
      for (int j=0; j<inner; j++) {
        I2.Direct(i,j) = I2(2*i,2*i,2*j,2*j);
        I2.Exchange(i,j) = I2(2*i,2*j,2*j,2*i);
    }

  } // commrank=0

#ifndef SERIAL
  mpi::broadcast(world, I1, 0);

  long intdim = I2memory;
  long  maxint = 26843540; //mpi cannot transfer more than these number of doubles
  long maxIter = intdim/maxint;

  world.barrier();
  for (int i=0; i<maxIter; i++) {
    mpi::broadcast(world, &I2.store[i*maxint], maxint, 0);
    world.barrier();
  }
  mpi::broadcast(world, &I2.store[maxIter*maxint], I2memory - maxIter*maxint, 0);
  world.barrier();

  mpi::broadcast(world, I2.maxEntry, 0);
  mpi::broadcast(world, I2.Direct, 0);
  mpi::broadcast(world, I2.Exchange, 0);
  mpi::broadcast(world, I2.zero, 0);
  mpi::broadcast(world, coreE, 0);
#endif

  Determinant::EffDetLen = (norbs) / 64 + 1;
  Determinant::norbs = norbs;
  Determinant::nalpha = nalpha;
  Determinant::nbeta = nbeta;

  //initialize the heatbath integrals
  std::vector<int> allorbs;
  std::vector<int> innerorbs;
  for (int i = 0; i < norbs; i++)
    allorbs.push_back(i);
  for (int i = 0; i < inner; i++)
    innerorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBath I2HBCAS(1.e-10);

  if (commrank == 0) {
    //if (schd.nciAct > 0) I2HB.constructClass(innerorbs, I2, I1, 0, norbs);
    //else I2HB.constructClass(allorbs, I2, I1, 0, norbs);
    I2HB.constructClass(innerorbs, I2, I1, 0, norbs);
    if (schd.nciCore > 0 || schd.nciAct > 0) I2HBCAS.constructClass(allorbs, I2, I1, schd.nciCore, schd.nciAct, true);
  }
  I2hb.constructClass(norbs, I2HB, 0);
  if (schd.nciAct > 0 || schd.nciAct > 0) I2hbCAS.constructClass(norbs, I2HBCAS, 1);

} // end readIntegrals


//=============================================================================
void readIntegralsHDF5AndInitializeDeterminantStaticVariables(string fcidump) {
//-----------------------------------------------------------------------------
    /*!
    Read fcidump file and populate "I1, I2, coreE, nelec, norbs"
    NB: I2 assumed to be 8-fold symmetric, irreps not implemented

    :Inputs:

        string fcidump:
            Name of the FCIDUMP file
    */
//-----------------------------------------------------------------------------
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  vector<int> irrep;
  int nelec, sz, norbs, nalpha, nbeta;
  hid_t file = (-1), dataset_header, dataset_hcore, dataset_eri, dataset_iiii, dataset_iiiv, dataset_iviv, dataset_iivv, dataset_energy_core ;  /* identifiers */
  herr_t status;

#ifndef SERIAL
  if (commrank == 0) {
#endif
    cout << "Reading integrals\n";
    norbs = -1;
    nelec = -1;
    sz = -1;
    H5E_BEGIN_TRY {
    file = H5Fopen(fcidump.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    } H5E_END_TRY
 
    if (file < 0) {
      if (commrank == 0) cout << "FCIDUMP not found!" << endl;
      exit(0);
    }

    int header[3];
    header[0] = 0;  //nelec
    header[1] = 0;  //norbs
    header[2] = 0;  //ms2
    dataset_header = H5Dopen(file, "/header", H5P_DEFAULT);
    status = H5Dread(dataset_header, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, header);
    I2.ksym = false;
    bool startScaling = false;
    nelec = header[0]; norbs = header[1]; sz = header[2];
    if (norbs == -1 || nelec == -1 || sz == -1) {
      std::cout << "could not read the norbs or nelec or MS2"<<std::endl;
      exit(0);
    }
    nalpha = nelec/2 + sz/2;
    nbeta = nelec - nalpha;
    irrep.resize(norbs);

#ifndef SERIAL
  } // commrank=0


  mpi::broadcast(world, nalpha, 0);
  mpi::broadcast(world, nbeta, 0);
  mpi::broadcast(world, nelec, 0);
  mpi::broadcast(world, norbs, 0);
  mpi::broadcast(world, irrep, 0);
  mpi::broadcast(world, I2.ksym, 0);
#endif

  long npair = norbs*(norbs+1)/2;
  if (I2.ksym) npair = norbs*norbs;
  I2.norbs = norbs;
  I2.npair = npair;
  int inner = norbs;
  if (schd.nciAct > 0) inner = schd.nciCore + schd.nciAct;
  I2.inner = inner;
  //innerDetLen = DetLen;
  int virt = norbs - inner;
  I2.virt = virt;
  //size_t nvirtpair = virt*(virt+1)/2;
  size_t nii = inner*(inner+1)/2;
  size_t niv = inner*virt;
  size_t nvv = virt*(virt+1)/2;
  size_t niiii = nii*(nii+1)/2;
  size_t niiiv = nii*niv;
  size_t niviv = niv*(niv+1)/2;
  size_t niivv = nii*nvv;
  I2.nii = nii;
  I2.niv = niv;
  I2.nvv = nvv;
  I2.niiii = niiii;
  I2.niiiv = niiiv;
  I2.niviv = niviv;
  I2.niivv = niivv;
  size_t I2memory = niiii + niiiv + niviv + niivv;
  //size_t I2memory = npair*(npair+1)/2 - nvirtpair*(nvirtpair+1)/2; //memory in bytes

#ifndef SERIAL
  world.barrier();
#endif

  int2Segment.truncate((I2memory)*sizeof(double));
  regionInt2 = boost::interprocess::mapped_region{int2Segment, boost::interprocess::read_write};
  memset(regionInt2.get_address(), 0., (I2memory)*sizeof(double));

#ifndef SERIAL
  world.barrier();
#endif

  I2.store = static_cast<double*>(regionInt2.get_address());

  if (commrank == 0) {

    I1.store.clear();
    I1.store.resize(2*norbs*(2*norbs),0.0); I1.norbs = 2*norbs;
    coreE = 0.0;

    double *hcore = new double[norbs * norbs];
    for (int i = 0; i < norbs; i++) 
      for (int j = 0; j < norbs; j++)
        hcore[i * norbs + j] = 0.;
    dataset_hcore = H5Dopen(file, "/hcore", H5P_DEFAULT);
    status = H5Dread(dataset_hcore, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, hcore);
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++) {
        double integral = hcore[i * norbs + j];
        I1(2*i, 2*j) = integral;
        I1(2*i+1, 2*j+1) = integral;
        I1(2*j, 2*i) = integral;
        I1(2*j + 1, 2*i + 1) = integral;
      }
    }
    delete [] hcore;

    //assuming 8-fold symmetry
    H5E_BEGIN_TRY {
      dataset_eri = H5Dopen(file, "/eri", H5P_DEFAULT);
    } H5E_END_TRY
    if (dataset_eri > 0) {
      unsigned int eri_size = npair * (npair + 1) / 2;
      double *eri = new double[eri_size];
      for (unsigned int i = 0; i < eri_size; i++)
        eri[i] = 0.;
      status = H5Dread(dataset_eri, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, eri);
      unsigned int ij = 0;
      unsigned int ijkl = 0;
      for (int i = 0; i < norbs; i++) {
        for (int j = 0; j < i + 1; j++) {
          int kl = 0;
          for (int k = 0; k < i + 1; k++) {
            for (int l = 0; l < k + 1; l++) {
              int n = 0;
              if (i >= inner) n++;
              if (j >= inner) n++;
              if (k >= inner) n++;
              if (l >= inner) n++;
              if (ij >= kl) {
                if (n < 3) I2(2*i, 2*j, 2*k, 2*l) = eri[ijkl];
                //I2(2*i, 2*j, 2*k, 2*l) = eri[ijkl];
                ijkl++;
              }
              kl++;
            }
          }
          ij++;
        }
      }
      delete [] eri;
    }
    else {
      dataset_iiii = H5Dopen(file, "/iiii", H5P_DEFAULT);
      double *iiii = new double[niiii];
      for (unsigned int i = 0; i < niiii; i++)
        iiii[i] = 0.;
      status = H5Dread(dataset_iiii, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, iiii);
      unsigned int ij = 0;
      unsigned int ijkl = 0;
      for (int i = 0; i < inner; i++) {
        for (int j = 0; j < i + 1; j++) {
          int kl = 0;
          for (int k = 0; k < i + 1; k++) {
            for (int l = 0; l < k + 1; l++) {
              if (ij >= kl) {
                I2(2*i, 2*j, 2*k, 2*l) = iiii[ijkl];
                //I2(2*i, 2*j, 2*k, 2*l) = eri[ijkl];
                ijkl++;
              }
              kl++;
            }
          }
          ij++;
        }
      }
      delete [] iiii;
      
      dataset_iiiv = H5Dopen(file, "/iiiv", H5P_DEFAULT);
      double *iiiv = new double[niiiv];
      for (size_t i = 0; i < niiiv; i++)
        iiiv[i] = 0.;
      status = H5Dread(dataset_iiiv, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, iiiv);
      for (size_t iv = 0; iv < niv; iv++) {
        size_t i = iv / inner + inner; 
        size_t j = iv % inner;
        for (size_t k = 0; k < inner; k++) {
          for (size_t l = 0; l <= k; l++) {
            size_t ii = k*(k+1)/2 + l;
            I2(2*i, 2*j, 2*k, 2*l) = iiiv[iv*nii + ii];
          }
        }
      }
      delete [] iiiv;
      
      dataset_iivv = H5Dopen(file, "/iivv", H5P_DEFAULT);
      double *iivv = new double[niivv];
      for (size_t i = 0; i < niivv; i++)
        iivv[i] = 0.;
      status = H5Dread(dataset_iivv, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, iivv);
      for (size_t i = inner; i < norbs; i++) {
        for (size_t j = inner; j <= i; j++) {
          size_t vv = (i-inner) * (i-inner+1) / 2 + (j-inner);
          for (size_t k = 0; k < inner; k++) {
            for (size_t l = 0; l <= k; l++) {
              size_t ii = k*(k+1)/2 + l;
              I2(2*i, 2*j, 2*k, 2*l) = iivv[vv*nii + ii];
            }
          }
        }
      }
      delete [] iivv;
      
      dataset_iviv = H5Dopen(file, "/iviv", H5P_DEFAULT);
      double *iviv = new double[niviv];
      for (size_t i = 0; i < niviv; i++)
        iviv[i] = 0.;
      status = H5Dread(dataset_iviv, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, iviv);
      for (size_t iv1 = 0; iv1 < niv; iv1++) {
        size_t i = iv1 / inner + inner; 
        size_t j = iv1 % inner;
        for (size_t iv2 = 0; iv2 <= iv1; iv2++) {
          size_t k = iv2 / inner + inner; 
          size_t l = iv2 % inner;
          I2(2*i, 2*j, 2*k, 2*l) = iviv[iv1*(iv1+1)/2 + iv2];
        }
      }
      delete [] iviv;
    }

    double energy_core[1];
    energy_core[0] = 0.;
    dataset_energy_core = H5Dopen(file, "/energy_core", H5P_DEFAULT);
    status = H5Dread(dataset_energy_core, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, energy_core);
    coreE = energy_core[0];

    status = H5Fclose(file);

    //exit(0);
    I2.maxEntry = *std::max_element(&I2.store[0], &I2.store[0]+I2memory,myfn);
    I2.Direct = MatrixXd::Zero(norbs, norbs); I2.Direct *= 0.;
    I2.Exchange = MatrixXd::Zero(norbs, norbs); I2.Exchange *= 0.;

    for (int i=0; i<inner; i++)
      for (int j=0; j<inner; j++) {
        I2.Direct(i,j) = I2(2*i,2*i,2*j,2*j);
        I2.Exchange(i,j) = I2(2*i,2*j,2*j,2*i);
    }
    cout << "Finished reading integrals\n";
  } // commrank=0

#ifndef SERIAL
  mpi::broadcast(world, I1, 0);

  long intdim = I2memory;
  long  maxint = 26843540; //mpi cannot transfer more than these number of doubles
  long maxIter = intdim/maxint;

  world.barrier();
  for (int i=0; i<maxIter; i++) {
    mpi::broadcast(world, &I2.store[i*maxint], maxint, 0);
    world.barrier();
  }
  mpi::broadcast(world, &I2.store[maxIter*maxint], I2memory - maxIter*maxint, 0);
  world.barrier();

  mpi::broadcast(world, I2.maxEntry, 0);
  mpi::broadcast(world, I2.Direct, 0);
  mpi::broadcast(world, I2.Exchange, 0);
  mpi::broadcast(world, I2.zero, 0);
  mpi::broadcast(world, coreE, 0);
#endif

  Determinant::EffDetLen = (norbs) / 64 + 1;
  Determinant::norbs = norbs;
  Determinant::nalpha = nalpha;
  Determinant::nbeta = nbeta;

  //initialize the heatbath integrals
  std::vector<int> allorbs;
  std::vector<int> innerorbs;
  for (int i = 0; i < norbs; i++)
    allorbs.push_back(i);
  for (int i = 0; i < inner; i++)
    innerorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBath I2HBCAS(1.e-10);

  if (commrank == 0) {
    cout << "Starting heat bath integral construction\n";
    //if (schd.nciAct > 0) I2HB.constructClass(innerorbs, I2, I1, 0, norbs);
    //else I2HB.constructClass(allorbs, I2, I1, 0, norbs);
    I2HB.constructClass(innerorbs, I2, I1, 0, norbs);
    if (schd.nciCore > 0 || schd.nciAct > 0) I2HBCAS.constructClass(allorbs, I2, I1, schd.nciCore, schd.nciAct, true);
  }
  I2hb.constructClass(norbs, I2HB, 0);
  if (schd.nciAct > 0 || schd.nciAct > 0) I2hbCAS.constructClass(norbs, I2HBCAS, 1);
  if (commrank == 0) cout << "Finished heat bath integral construction\n";

} // end readIntegrals


void readDQMCIntegralsRG(string fcidump, int& norbs, int& nalpha, int& nbeta, double& ecore, MatrixXd& h1, MatrixXd& h1Mod, vector<Eigen::Map<MatrixXd>>& chol, vector<Eigen::Map<MatrixXd>>& cholMat, bool ghf) {
  int nelec, sz, nchol;
  hid_t file = (-1), dataset_header = (-1), dataset_energy_core = (-1);  
  herr_t status;

  H5E_BEGIN_TRY {
  file = H5Fopen(fcidump.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  } H5E_END_TRY
  if (file < 0) {
    if (commrank == 0) cout << "Cholesky integrals not found!" << endl;
    exit(1);
  }

  int header[4];
  for (int i = 0; i < 4; i++) header[i] = 0;
  
  H5E_BEGIN_TRY {
    dataset_header = H5Dopen(file, "/header", H5P_DEFAULT);
    status = H5Dread(dataset_header, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, header);
  } H5E_END_TRY
  if (dataset_header < 0) {
    if (commrank == 0) cout << "Header could not be read." << endl;
    exit(1);
  }
  
  nelec = header[0]; norbs = header[1]; sz = header[2]; nchol = header[3];
  nalpha = (nelec + sz)/2;
  nbeta = nelec - nalpha;

  // these shouldn't really be used anywhere in afqmc
  Determinant::EffDetLen = (norbs) / 64 + 1;
  Determinant::norbs = norbs;
  Determinant::nalpha = nalpha;
  Determinant::nbeta = nbeta;

  h1 = MatrixXd::Zero(norbs, norbs);
  readMat(h1, file, "/hcore"); 
  h1Mod = MatrixXd::Zero(norbs, norbs);
  readMat(h1Mod, file, "/hcore_mod"); 

  // read cholesky to shared memory
  size_t cholSize = nchol * norbs * norbs;
  double* cholSHM;
  MPI_Barrier(MPI_COMM_WORLD);
  readHDF5ToSHM(file, "/chol", cholSize, cholSHM, cholSHMName, cholSegment, cholRegion);
  MPI_Barrier(MPI_COMM_WORLD);

  // create eigen matrix maps to shared memory
  for (size_t n = 0; n < nchol; n++) {
    Eigen::Map<MatrixXd> cholMatMap(static_cast<double*>(cholSHM) + n * norbs * norbs, norbs, norbs);
    chol.push_back(cholMatMap);
  }

  Eigen::Map<MatrixXd> cholMatMap(static_cast<double*>(cholSHM), norbs * norbs, nchol);
  cholMat.push_back(cholMatMap);

  coreE = 0.;
  double energy_core[1];
  energy_core[0] = 0.;
  H5E_BEGIN_TRY {
    dataset_energy_core = H5Dopen(file, "/energy_core", H5P_DEFAULT);
    status = H5Dread(dataset_energy_core, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, energy_core);
  } H5E_END_TRY
  if (dataset_energy_core < 0) {
    if (commrank == 0) cout << "Core energy could not be read, setting to zero." << endl;
  }
  else {
    coreE = energy_core[0];
    ecore = energy_core[0];
  }

  status = H5Fclose(file);
} 


void readDQMCIntegralsU(string fcidump, int& norbs, int& nalpha, int& nbeta, double& ecore, std::array<MatrixXd, 2>& h1, std::array<MatrixXd, 2>& h1Mod, vector<std::array<Eigen::Map<MatrixXd>, 2>>& chol) {
  int nelec, sz, nchol;
  hid_t file = (-1), dataset_header = (-1), dataset_energy_core = (-1);  
  herr_t status;

  H5E_BEGIN_TRY {
  file = H5Fopen(fcidump.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  } H5E_END_TRY
  if (file < 0) {
    if (commrank == 0) cout << "Cholesky integrals not found!" << endl;
    exit(1);
  }

  int header[4];
  for (int i = 0; i < 4; i++) header[i] = 0;
  
  H5E_BEGIN_TRY {
    dataset_header = H5Dopen(file, "/header", H5P_DEFAULT);
    status = H5Dread(dataset_header, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, header);
  } H5E_END_TRY
  if (dataset_header < 0) {
    if (commrank == 0) cout << "Header could not be read." << endl;
    exit(1);
  }
  
  nelec = header[0]; norbs = header[1]; sz = header[2]; nchol = header[3];
  nalpha = (nelec + sz)/2;
  nbeta = nelec - nalpha;
 
  // these shouldn't really be used anywhere in afqmc
  Determinant::EffDetLen = (norbs) / 64 + 1;
  Determinant::norbs = norbs;
  Determinant::nalpha = nalpha;
  Determinant::nbeta = nbeta;

  h1[0] = MatrixXd::Zero(norbs, norbs);
  h1[1] = MatrixXd::Zero(norbs, norbs);
  readMat(h1[0], file, "/hcore_up"); 
  readMat(h1[1], file, "/hcore_dn"); 
  h1Mod[0] = MatrixXd::Zero(norbs, norbs);
  h1Mod[1] = MatrixXd::Zero(norbs, norbs);
  readMat(h1Mod[0], file, "/hcore_mod_up"); 
  readMat(h1Mod[1], file, "/hcore_mod_dn"); 

  // read cholesky to shared memory
  size_t cholSize = nchol * norbs * norbs;
  double* cholSHMUp;
  double* cholSHMDn;
  MPI_Barrier(MPI_COMM_WORLD);
  readHDF5ToSHM(file, "/chol_up", cholSize, cholSHMUp, cholSHMNameUp, cholSegmentUp, cholRegionUp);
  readHDF5ToSHM(file, "/chol_dn", cholSize, cholSHMDn, cholSHMNameDn, cholSegmentDn, cholRegionDn);
  MPI_Barrier(MPI_COMM_WORLD);

  // create eigen matrix maps to shared memory
  for (size_t n = 0; n < nchol; n++) {
    Eigen::Map<MatrixXd> cholMatUp(static_cast<double*>(cholSHMUp) + n * norbs * norbs, norbs, norbs);
    Eigen::Map<MatrixXd> cholMatDn(static_cast<double*>(cholSHMDn) + n * norbs * norbs, norbs, norbs);
    std::array<Eigen::Map<MatrixXd>, 2> cholMat = {cholMatUp, cholMatDn};
    chol.push_back(cholMat);
  }

  coreE = 0.;
  double energy_core[1];
  energy_core[0] = 0.;
  H5E_BEGIN_TRY {
    dataset_energy_core = H5Dopen(file, "/energy_core", H5P_DEFAULT);
    status = H5Dread(dataset_energy_core, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, energy_core);
  } H5E_END_TRY
  if (dataset_energy_core < 0) {
    if (commrank == 0) cout << "Core energy could not be read, setting to zero." << endl;
  }
  else {
    coreE = energy_core[0];
    ecore = energy_core[0];
  }

  status = H5Fclose(file);
} 


void readDQMCIntegralsSOC(string fcidump, int& norbs, int& nelec, double& ecore, MatrixXcd& h1, MatrixXcd& h1Mod, vector<Eigen::Map<MatrixXd>>& chol) {
  int nchol;
  hid_t file = (-1), dataset_header = (-1), dataset_energy_core = (-1);  
  herr_t status;

  H5E_BEGIN_TRY {
  file = H5Fopen(fcidump.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  } H5E_END_TRY
  if (file < 0) {
    if (commrank == 0) cout << "Cholesky integrals not found!" << endl;
    exit(1);
  }

  int header[3];
  for (int i = 0; i < 3; i++) header[i] = 0;
  
  H5E_BEGIN_TRY {
    dataset_header = H5Dopen(file, "/header", H5P_DEFAULT);
    status = H5Dread(dataset_header, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, header);
  } H5E_END_TRY
  if (dataset_header < 0) {
    if (commrank == 0) cout << "Header could not be read." << endl;
    exit(1);
  }
  nelec = header[0]; norbs = header[1]; nchol = header[2];
  
  Determinant::EffDetLen = (norbs) / 64 + 1;
  Determinant::norbs = norbs;
  Determinant::nalpha = 0;
  Determinant::nbeta = 0;

  h1 = MatrixXcd::Zero(2*norbs, 2*norbs);
  readMat(h1, file, "/hcore"); 
  h1Mod = MatrixXcd::Zero(2*norbs, 2*norbs);
  readMat(h1Mod, file, "/hcore_mod"); 
  
  // read cholesky to shared memory
  unsigned int cholSize = nchol * norbs * norbs;
  double* cholSHM;
  MPI_Barrier(MPI_COMM_WORLD);
  readHDF5ToSHM(file, "/chol", cholSize, cholSHM, cholSHMName, cholSegment, cholRegion);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // create eigen matrix maps to shared memory
  for (int n = 0; n < nchol; n++) {
    Eigen::Map<MatrixXd> cholMat(static_cast<double*>(cholSHM) + n * norbs * norbs, norbs, norbs);
    chol.push_back(cholMat);
  }

  coreE = 0.;
  double energy_core[1];
  energy_core[0] = 0.;
  H5E_BEGIN_TRY {
    dataset_energy_core = H5Dopen(file, "/energy_core", H5P_DEFAULT);
    status = H5Dread(dataset_energy_core, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, energy_core);
  } H5E_END_TRY
  if (dataset_energy_core < 0) {
    if (commrank == 0) cout << "Core energy could not be read, setting to zero." << endl;
  }
  else {
    coreE = energy_core[0];
    ecore = energy_core[0];
  }

  status = H5Fclose(file);
} 

//=============================================================================
int readNorbs(string fcidump) {
//-----------------------------------------------------------------------------
    /*!
    Finds the number of orbitals in the FCIDUMP file

    :Inputs:

        string fcidump:
            Name of the FCIDUMP file

    :Returns:

        int norbs:
            Number of orbitals in the FCIDUMP file
    */
//-----------------------------------------------------------------------------
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int norbs;
  if (commrank == 0) {
    ifstream dump(fcidump.c_str());
    vector<string> tok;
    string msg;

    std::getline(dump, msg);
    trim(msg);
    boost::split(tok, msg, is_any_of(", \t="), token_compress_on);

    if (boost::iequals(tok[0].substr(0,4),"&FCI"))
      if (boost::iequals(tok[1].substr(0,4), "NORB"))
        norbs = atoi(tok[2].c_str());
  }
#ifndef SERIAL
  mpi::broadcast(world, norbs, 0);
#endif
  return norbs;
} // end readNorbs


//=============================================================================
void twoIntHeatBathSHM::constructClass(int norbs, twoIntHeatBath& I2, bool cas) {
//-----------------------------------------------------------------------------
    /*!
    BM_description

    :Inputs:

        int norbs:
            Number of orbitals
        twoIntHeatBath& I2:
            Two-electron tensor of the Hamiltonian (output)
    */
//-----------------------------------------------------------------------------
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

  sameSpinPairExcitations = MatrixXd::Zero(norbs, norbs);
  oppositeSpinPairExcitations = MatrixXd::Zero(norbs, norbs);

  Singles = I2.Singles;
  if (commrank != 0) Singles.resize(2*norbs, 2*norbs);

  if (!cas) {
#ifndef SERIAL
  mpi::broadcast(world, &Singles(0,0), Singles.rows()*Singles.cols(), 0);
#endif
  }


  I2.Singles.resize(0,0);
  size_t memRequired = 0;
  size_t nonZeroSameSpinIntegrals = 0;
  size_t nonZeroOppositeSpinIntegrals = 0;
  size_t nonZeroSingleExcitationIntegrals = 0;

  if (commrank == 0) {
    std::map<std::pair<short,short>, std::multimap<float, std::pair<short,short>, compAbs > >::iterator it1 = I2.sameSpin.begin();
    for (;it1!= I2.sameSpin.end(); it1++)  nonZeroSameSpinIntegrals += it1->second.size();
      
    std::map<std::pair<short,short>, std::multimap<float, std::pair<short,short>, compAbs > >::iterator it2 = I2.oppositeSpin.begin();
    for (;it2!= I2.oppositeSpin.end(); it2++) nonZeroOppositeSpinIntegrals += it2->second.size();

    std::map<std::pair<short,short>, std::multimap<float, short, compAbs > >::iterator it3 = I2.singleIntegrals.begin();
    for (;it3!= I2.singleIntegrals.end(); it3++) nonZeroSingleExcitationIntegrals += it3->second.size();

    //total Memory required
    memRequired += nonZeroSameSpinIntegrals*(sizeof(float)+2*sizeof(short))+ ( (norbs*(norbs+1)/2+1)*sizeof(size_t));
    memRequired += nonZeroOppositeSpinIntegrals*(sizeof(float)+2*sizeof(short))+ ( (norbs*(norbs+1)/2+1)*sizeof(size_t));
    memRequired += nonZeroSingleExcitationIntegrals*(sizeof(float)+2*sizeof(short))+ ( (norbs*(norbs+1)/2+1)*sizeof(size_t));
  }

#ifndef SERIAL
  mpi::broadcast(world, memRequired, 0);
  mpi::broadcast(world, nonZeroSameSpinIntegrals, 0);
  mpi::broadcast(world, nonZeroOppositeSpinIntegrals, 0);
  mpi::broadcast(world, nonZeroSingleExcitationIntegrals, 0);
  world.barrier();
#endif

  if (!cas) {
    int2SHMSegment.truncate(memRequired);
    regionInt2SHM = boost::interprocess::mapped_region{int2SHMSegment, boost::interprocess::read_write};
    memset(regionInt2SHM.get_address(), 0., memRequired);
  }
  else {
    int2SHMCASSegment.truncate(memRequired);
    regionInt2SHMCAS = boost::interprocess::mapped_region{int2SHMCASSegment, boost::interprocess::read_write};
    memset(regionInt2SHMCAS.get_address(), 0., memRequired);
  }

#ifndef SERIAL
  world.barrier();
#endif

  char* startAddress;
  if (!cas) startAddress = (char*)(regionInt2SHM.get_address());
  else startAddress = (char*)(regionInt2SHMCAS.get_address());
  sameSpinIntegrals           = (float*)(startAddress);
  startingIndicesSameSpin     = (size_t*)(startAddress
                              + nonZeroSameSpinIntegrals*sizeof(float));
  sameSpinPairs               = (short*)(startAddress
                              + nonZeroSameSpinIntegrals*sizeof(float)
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t));

  oppositeSpinIntegrals       = (float*)(startAddress
                              + nonZeroSameSpinIntegrals*(sizeof(float)+2*sizeof(short))
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t));
  startingIndicesOppositeSpin = (size_t*)(startAddress
                              + nonZeroOppositeSpinIntegrals*sizeof(float)
                              + nonZeroSameSpinIntegrals*(sizeof(float)+2*sizeof(short))
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t));
  oppositeSpinPairs             = (short*)(startAddress
                              + nonZeroOppositeSpinIntegrals*sizeof(float)
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t)
                              + nonZeroSameSpinIntegrals*(sizeof(float)+2*sizeof(short))
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t));

  singleIntegrals             = (float*)(startAddress
                              + nonZeroOppositeSpinIntegrals*sizeof(float)
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t)
                              + nonZeroSameSpinIntegrals*(sizeof(float)+2*sizeof(short))
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t)
			      + nonZeroOppositeSpinIntegrals*(2*sizeof(short)));
  startingIndicesSingleIntegrals = (size_t*)(startAddress
                              + nonZeroOppositeSpinIntegrals*sizeof(float)
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t)
                              + nonZeroSameSpinIntegrals*(sizeof(float)+2*sizeof(short))
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t)
			      + nonZeroOppositeSpinIntegrals*(2*sizeof(short))
			      + nonZeroSingleExcitationIntegrals*sizeof(float));
  singleIntegralsPairs        = (short*)(startAddress
                              + nonZeroOppositeSpinIntegrals*sizeof(float)
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t)
                              + nonZeroSameSpinIntegrals*(sizeof(float)+2*sizeof(short))
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t)
			      + nonZeroOppositeSpinIntegrals*(2*sizeof(short))
			      + nonZeroSingleExcitationIntegrals*sizeof(float)
                              + (norbs*(norbs+1)/2+1)*sizeof(size_t));

  if (commrank == 0) {
    startingIndicesSameSpin[0] = 0;
    size_t index = 0, pairIter = 1;
    for (int i=0; i<norbs; i++)
      for (int j=0; j<=i; j++) {
        std::map<std::pair<short,short>, std::multimap<float, std::pair<short,short>, compAbs > >::iterator it1 = I2.sameSpin.find( std::pair<short,short>(i,j));

        if (it1 != I2.sameSpin.end()) {
          for (std::multimap<float, std::pair<short,short>,compAbs >::reverse_iterator it=it1->second.rbegin(); it!=it1->second.rend(); it++) {
            sameSpinIntegrals[index] = it->first;
            sameSpinPairs[2*index] = it->second.first;
            sameSpinPairs[2*index+1] = it->second.second;
	    sameSpinPairExcitations(it->second.first, it->second.second) += abs(it->first); 

            index++;
          }
        }
        startingIndicesSameSpin[pairIter] = index;

        pairIter++;
    }
    I2.sameSpin.clear();

    startingIndicesOppositeSpin[0] = 0;
    index = 0; pairIter = 1;
    for (int i=0; i<norbs; i++)
      for (int j=0; j<=i; j++) {
        std::map<std::pair<short,short>, std::multimap<float, std::pair<short,short>, compAbs > >::iterator it1 = I2.oppositeSpin.find( std::pair<short,short>(i,j));

        if (it1 != I2.oppositeSpin.end()) {
          for (std::multimap<float, std::pair<short,short>,compAbs >::reverse_iterator it=it1->second.rbegin(); it!=it1->second.rend(); it++) {
            oppositeSpinIntegrals[index] = it->first;
            oppositeSpinPairs[2*index] = it->second.first;
            oppositeSpinPairs[2*index+1] = it->second.second;
	    oppositeSpinPairExcitations(it->second.first, it->second.second) += abs(it->first); 
            index++;
          }
        }
        startingIndicesOppositeSpin[pairIter] = index;
        pairIter++;
    }
    I2.oppositeSpin.clear();

    startingIndicesSingleIntegrals[0] = 0;
    index = 0; pairIter = 1;
    for (int i=0; i<norbs; i++)
      for (int j=0; j<=i; j++) {
        std::map<std::pair<short,short>, std::multimap<float, short, compAbs > >::iterator it1 = I2.singleIntegrals.find( std::pair<short,short>(i,j));

        if (it1 != I2.singleIntegrals.end()) {
          for (std::multimap<float, short,compAbs >::reverse_iterator it=it1->second.rbegin(); it!=it1->second.rend(); it++) {
            singleIntegrals[index] = it->first;
            singleIntegralsPairs[2*index] = it->second;
            index++;
          }
        }
        startingIndicesSingleIntegrals[pairIter] = index;
        pairIter++;
    }
    I2.singleIntegrals.clear();



  } // commrank=0

  long intdim = memRequired;
  long  maxint = 26843540; //mpi cannot transfer more than these number of doubles
  long maxIter = intdim/maxint;
#ifndef SERIAL
  world.barrier();
  char* shrdMem = static_cast<char*>(startAddress);
  for (int i=0; i<maxIter; i++) {
    mpi::broadcast(world, shrdMem+i*maxint, maxint, 0);
    world.barrier();
  }
  mpi::broadcast(world, shrdMem+(maxIter)*maxint, memRequired - maxIter*maxint, 0);
  world.barrier();
#endif

#ifndef SERIAL
  mpi::broadcast(world, &sameSpinPairExcitations(0,0), sameSpinPairExcitations.rows()*
        sameSpinPairExcitations.cols(), 0);
  mpi::broadcast(world, &oppositeSpinPairExcitations(0,0), oppositeSpinPairExcitations.rows()*
        oppositeSpinPairExcitations.cols(), 0);
#endif

} // end twoIntHeatBathSHM::constructClass


void twoIntHeatBathSHM::getIntegralArray(int i, int j, const float* &integrals,
                                         const short* &orbIndices, size_t& numIntegrals) const {
  int I = i / 2, J = j / 2;
  int X = max(I, J), Y = min(I, J);

  int pairIndex     = X * (X + 1) / 2 + Y;
  size_t start      = i % 2 == j % 2 ? I2hb.startingIndicesSameSpin[pairIndex] : I2hb.startingIndicesOppositeSpin[pairIndex];
  size_t end        = i % 2 == j % 2 ? I2hb.startingIndicesSameSpin[pairIndex + 1] : I2hb.startingIndicesOppositeSpin[pairIndex + 1];
  integrals  = i % 2 == j % 2 ? I2hb.sameSpinIntegrals+start : I2hb.oppositeSpinIntegrals+start;
  orbIndices = i % 2 == j % 2 ? I2hb.sameSpinPairs+2*start : I2hb.oppositeSpinPairs+2*start;
  numIntegrals = end-start;
}

void twoIntHeatBathSHM::getIntegralArrayCAS(int i, int j, const float* &integrals,
                                         const short* &orbIndices, size_t& numIntegrals) const {
  int I = i / 2, J = j / 2;
  int X = max(I, J), Y = min(I, J);

  int pairIndex     = X * (X + 1) / 2 + Y;
  size_t start      = i % 2 == j % 2 ? I2hbCAS.startingIndicesSameSpin[pairIndex] : I2hbCAS.startingIndicesOppositeSpin[pairIndex];
  size_t end        = i % 2 == j % 2 ? I2hbCAS.startingIndicesSameSpin[pairIndex + 1] : I2hbCAS.startingIndicesOppositeSpin[pairIndex + 1];
  integrals  = i % 2 == j % 2 ? I2hbCAS.sameSpinIntegrals+start : I2hbCAS.oppositeSpinIntegrals+start;
  orbIndices = i % 2 == j % 2 ? I2hbCAS.sameSpinPairs+2*start : I2hbCAS.oppositeSpinPairs+2*start;
  numIntegrals = end-start;
}
