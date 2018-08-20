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
#include "string.h"
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include "math.h"
#include "boost/format.hpp"
#include <fstream>
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"
#include "global.h"

using namespace boost;

#ifndef Complex
bool myfn(CItype i, CItype j) { return fabs(i)<fabs(j); }
#else
bool myfn(CItype i, CItype j) { return std::abs(i) < std::abs(j); }
#endif

//=============================================================================
void readIntegrals(string fcidump, twoInt& I2, oneInt& I1, int& nelec, int& norbs, double& coreE, std::vector<int>& irrep) {
//-----------------------------------------------------------------------------
    /*!
    Read FCIDUMP file and populate "I1, I2, coreE, nelec, norbs, irrep"

    :Inputs:

        string fcidump:
            Name of the FCIDUMP file
        twoInt& I2:
            Two-electron tensor of the Hamiltonian (output)
        oneInt& I1:
            One-electron tensor of the Hamiltonian (output)
        int& nelec:
            Number of electrons (output)
        int& norbs:
            Number of orbitals (output)
        double& coreE:
            The core energy (output)
        std::vector<int>& irrep:
            Irrep of the orbitals (output)
    */
//-----------------------------------------------------------------------------
#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  ifstream dump(fcidump.c_str());
  if (!dump.good()) {
    pout << "Integral file "<<fcidump<<" does not exist!"<<endl;
    exit(0);
  }

  if (commrank == 0) {
    I2.ksym = false;
    bool startScaling = false;
    norbs = -1;
    nelec = -1;

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
        } 
        else if (boost::iequals(tok[0].substr(0,4),"ISYM")) continue;
        else if (boost::iequals(tok[0].substr(0,4),"KSYM")) I2.ksym = true;
        else if (boost::iequals(tok[0].substr(0,6),"ORBSYM")) {
          for (int i=1;i<tok.size(); i++)
            irrep.push_back(atoi(tok[i].c_str()));
        } 
        else {
          for (int i=0;i<tok.size(); i++)
            irrep.push_back(atoi(tok[i].c_str()));
        }
        index += 1;
      }
    } // while

    if (norbs == -1 || nelec == -1) {
      std::cout << "could not read the norbs or nelec"<<std::endl;
      exit(0);
    }
    irrep.resize(norbs);
  } // commrank=0

#ifndef SERIAL
  mpi::broadcast(world, nelec, 0);
  mpi::broadcast(world, norbs, 0);
  mpi::broadcast(world, irrep, 0);
  mpi::broadcast(world, I2.ksym, 0);
#endif

  //long npair = norbs*(norbs+1)/2;
  //size_t I2memory = npair*(npair+1)/2; //memory in bytes
  long npair = norbs*norbs;
  I2.norbs = norbs;
  size_t I2memory = npair*(npair+1)/2;
#ifndef SERIAL
  world.barrier();
#endif

  int2Segment.truncate((I2memory)*sizeof(std::complex<double>));
  regionInt2 = boost::interprocess::mapped_region{int2Segment, boost::interprocess::read_write};
  memset(regionInt2.get_address(), 0., (I2memory)*sizeof(std::complex<double>));

#ifndef SERIAL
  world.barrier();
#endif

  //I2.store = static_cast<double*>(regionInt2.get_address());
  I2.store = static_cast<CItype*>(regionInt2.get_address());
  if (commrank == 0) {
    I1.store.clear();
    I1.store.resize(norbs*norbs,0.0); I1.norbs = norbs;
    coreE = 0.0;

    vector<string> tok;
    string msg;
    while(!dump.eof()) {
      std::getline(dump, msg);
      trim(msg);
      boost::split(tok, msg, is_any_of(", \t()"), token_compress_on);
      //if (tok.size() != 5) continue;
      if (tok.size() != 7) continue;

      //double integral = atof(tok[0].c_str());int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str()), c=atoi(tok[3].c_str()), d=atoi(tok[4].c_str());
      CItype integral = std::complex<double>(atof(tok[1].c_str()), atof(tok[2].c_str())); 
      int a=atoi(tok[3].c_str()), b=atoi(tok[4].c_str()), c=atoi(tok[5].c_str()), d=atoi(tok[6].c_str());
      if(a==b && b==c && c==d && d==0)
        coreE = integral.real();
      else if (b==c && c==d && d==0)
        continue;//orbital energy
      else if (c==d&&d==0)
        I1(a-1,b-1) = integral;
      else
        I2(a-1,b-1,c-1,d-1) = I2(c-1,d-1,a-1,b-1)= integral;
    } // while
    
    I2.maxEntry = *std::max_element(&I2.store[0], &I2.store[0]+I2memory, myfn);
    I2.Direct = Matrix<std::complex<double>,-1,-1>::Zero(norbs, norbs); I2.Direct *= 0.;
    I2.Exchange = Matrix<std::complex<double>,-1,-1>::Zero(norbs, norbs); I2.Exchange *= 0.;

    for (int i=0; i<norbs; i++)
      for (int j=0; j<norbs; j++) {
        I2.Direct(i,j) = I2(i,i,j,j);
        I2.Exchange(i,j) = I2(i,j,j,i);
    }
  } // commrank=0

#ifndef SERIAL
  mpi::broadcast(world, I1, 0);

  long intdim = I2memory;
  long  maxint = 26843540; //mpi cannot transfer more than these number of doubles
  long maxIter = intdim/maxint;

  world.barrier();
  for (int i=0; i<maxIter; i++) {
    cout << i << " " << maxIter << endl;
    mpi::broadcast(world, &I2.store[i * maxint], maxint, 0);
    world.barrier();
  }
  mpi::broadcast(world, &I2.store[(maxIter)*maxint],
                 I2memory - maxIter * maxint, 0);
  world.barrier();

  mpi::broadcast(world, I2.maxEntry, 0);
  mpi::broadcast(world, I2.Direct, 0);
  mpi::broadcast(world, I2.Exchange, 0);
  mpi::broadcast(world, I2.zero, 0);
  mpi::broadcast(world, coreE, 0);
#endif
} // end readIntegrals



//=============================================================================
void twoIntHeatBathSHM::constructClass(int norbs, twoIntHeatBath& I2) {
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

  Singles = I2.Singles;
  if (commrank != 0) Singles.resize(norbs, norbs);
#ifndef SERIAL
  MPI::COMM_WORLD.Bcast(&Singles(0,0), Singles.rows()*Singles.cols(), MPI_DOUBLE, 0);
#endif

  I2.Singles.resize(0,0);
  size_t memRequired = 0;
  size_t nonZeroIntegrals = 0;
  //size_t nonZeroSameSpinIntegrals = 0;
  //size_t nonZeroOppositeSpinIntegrals = 0;

  if (commrank == 0) {
    //convert to CItype
    std::map<std::pair<short,short>, std::multimap<CItype, std::pair<short,short>, compAbs > >::iterator it1 = I2.integral.begin();
    for (;it1!= I2.integral.end(); it1++) nonZeroIntegrals += it1->second.size();

    //std::map<std::pair<short,short>, std::multimap<CItype, std::pair<short,short>, compAbs > >::iterator it2 = I2.oppositeSpin.begin();
    //for (;it2!= I2.oppositeSpin.end(); it2++) nonZeroOppositeSpinIntegrals += it2->second.size();

    //total Memory required, have to think of it carefully
    memRequired += nonZeroIntegrals*(sizeof(std::complex<double>)+2*sizeof(short))+ ( (norbs*(norbs+1)/2+1)*sizeof(size_t));
    //memRequired += nonZeroOppositeSpinIntegrals*(sizeof(float)+2*sizeof(short))+ ( (norbs*(norbs+1)/2+1)*sizeof(size_t));
  }

#ifndef SERIAL
  mpi::broadcast(world, memRequired, 0);
  mpi::broadcast(world, nonZeroIntegrals, 0);
  //mpi::broadcast(world, nonZeroSameSpinIntegrals, 0);
  //mpi::broadcast(world, nonZeroOppositeSpinIntegrals, 0);
  world.barrier();
#endif

  int2SHMSegment.truncate(memRequired);
  regionInt2SHM = boost::interprocess::mapped_region{int2SHMSegment, boost::interprocess::read_write};
  memset(regionInt2SHM.get_address(), 0., memRequired);

#ifndef SERIAL
  world.barrier();
#endif

  char* startAddress = (char*)(regionInt2SHM.get_address());
  //sameSpinIntegrals           = (float*)(startAddress);
  //startingIndicesSameSpin     = (size_t*)(startAddress
  //                            + nonZeroSameSpinIntegrals*sizeof(float));
  //sameSpinPairs               = (short*)(startAddress
  //                            + nonZeroSameSpinIntegrals*sizeof(float)
  //                            + (norbs*(norbs+1)/2+1)*sizeof(size_t));
  //oppositeSpinIntegrals       = (float*)(startAddress
  //                            + nonZeroSameSpinIntegrals*(sizeof(float)+2*sizeof(short))
  //                            + (norbs*(norbs+1)/2+1)*sizeof(size_t));
  //startingIndicesOppositeSpin = (size_t*)(startAddress
  //                            + nonZeroOppositeSpinIntegrals*sizeof(float)
  //                            + nonZeroSameSpinIntegrals*(sizeof(float)+2*sizeof(short))
  //                            + (norbs*(norbs+1)/2+1)*sizeof(size_t));
  //oppositeSpinPairs             = (short*)(startAddress
  //                            + nonZeroOppositeSpinIntegrals*sizeof(float)
  //                            + (norbs*(norbs+1)/2+1)*sizeof(size_t)
  //                            + nonZeroSameSpinIntegrals*(sizeof(float)+2*sizeof(short))
  //                            + (norbs*(norbs+1)/2+1)*sizeof(size_t));
  integrals                     = (std::complex<double>*)(startAddress);
  startingIndicesIntegrals      = (size_t*)(startAddress
                                + nonZeroIntegrals*sizeof(std::complex<double>));
  pairs                         = (short*)(startAddress
                                + nonZeroIntegrals*sizeof(std::complex<double>)
                                + norbs*norbs*sizeof(size_t));                            

  if (commrank == 0) {
    //startingIndicesSameSpin[0] = 0;
    startingIndicesIntegrals[0] = 0;
    size_t index = 0, pairIter = 1;
    for (int i=0; i<norbs; i++)
      for (int j=0; j<=i; j++) {
        //convert to CItype
        std::map<std::pair<short,short>, std::multimap<CItype, std::pair<short,short>, compAbs > >::iterator it1 = I2.integral.find( std::pair<short,short>(i,j));

        if (it1 != I2.integral.end()) {
          for (std::multimap<std::complex<double>, std::pair<short,short>,compAbs >::reverse_iterator it=it1->second.rbegin(); it!=it1->second.rend(); it++) {
            //sameSpinIntegrals[index] = it->first;
            //sameSpinPairs[2*index] = it->second.first;
            //sameSpinPairs[2*index+1] = it->second.second;
            integrals[index] = it->first;
            pairs[2*index] = it->second.first;
            pairs[2*index+1] = it->second.second;
            index++;
          }
        }
        //startingIndicesSameSpin[pairIter] = index;
        startingIndicesIntegrals[pairIter] = index;
        pairIter++;
    }
    I2.integral.clear();

    //startingIndicesOppositeSpin[0] = 0;
    //index = 0; pairIter = 1;
    //for (int i=0; i<norbs; i++)
    //  for (int j=0; j<=i; j++) {
    //    //convert to CItype
    //    std::map<std::pair<short,short>, std::multimap<CItype, std::pair<short,short>, compAbs > >::iterator it1 = I2.oppositeSpin.find( std::pair<short,short>(i,j));
//
    //    if (it1 != I2.oppositeSpin.end()) {
    //      for (std::multimap<float, std::pair<short,short>,compAbs >::reverse_iterator it=it1->second.rbegin(); it!=it1->second.rend(); it++) {
    //        oppositeSpinIntegrals[index] = it->first;
    //        oppositeSpinPairs[2*index] = it->second.first;
    //        oppositeSpinPairs[2*index+1] = it->second.second;
    //        index++;
    //      }
    //    }
    //    startingIndicesOppositeSpin[pairIter] = index;
    //    pairIter++;
    //}
    //I2.oppositeSpin.clear();
  } // commrank=0

  long intdim = memRequired;
  long  maxint = 26843540; //mpi cannot transfer more than these number of doubles
  long maxIter = intdim/maxint;
#ifndef SERIAL
  world.barrier();
  char* shrdMem = static_cast<char*>(startAddress);
  for (int i=0; i<maxIter; i++) {
    MPI::COMM_WORLD.Bcast(shrdMem+i*maxint, maxint, MPI_CHAR, 0);
    world.barrier();
  }
  MPI::COMM_WORLD.Bcast(shrdMem+(maxIter)*maxint, memRequired - maxIter*maxint, MPI_CHAR, 0);
  world.barrier();
#endif
} // end twoIntHeatBathSHM::constructClass



#ifdef Complex
//=============================================================================
void readSOCIntegrals(oneInt& I1, int norbs, string fileprefix) {
//-----------------------------------------------------------------------------
    /*!
    Read SOC integrals from files, to be put in "I1"

    :Inputs:

        oneInt& I1:
            One-electron tensor of the Hamiltonian (output)
        int norbs:
            Number of orbitals
        string fileprefix:
            Basename of the SOC integral files
    */
//-----------------------------------------------------------------------------
  if (commrank == 0) {
    vector<string> tok;
    string msg;

    //Read SOC.X
    {
      ifstream dump(str(boost::format("%s.X") % fileprefix));
      int N;
      dump >> N;
      if (N != norbs/2) {
        cout << "number of orbitals in SOC.X should be equal to norbs in the input file."<<endl;
        cout << N <<" != "<<norbs<<endl;
        exit(0);
      }

      //I1soc[1].store.resize(N*(N+1)/2, 0.0);
      while(!dump.eof()) {
        std::getline(dump, msg);
        trim(msg);
        boost::split(tok, msg, is_any_of(", \t="), token_compress_on);
        if (tok.size() != 3) continue;

        double integral = atof(tok[0].c_str());
        int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str());
        //I1(2*(a-1), 2*(b-1)+1) += std::complex<double>(0,integral/2.);  //alpha beta
        //I1(2*(a-1)+1, 2*(b-1)) += std::complex<double>(0,integral/2.);  //beta alpha
        I1(2*(a-1), 2*(b-1)+1) += std::complex<double>(0,-integral/2.);  //alpha beta
        I1(2*(a-1)+1, 2*(b-1)) += std::complex<double>(0,-integral/2.);  //beta alpha
      }
    }

    //Read SOC.Y
    {
      ifstream dump(str(boost::format("%s.Y") % fileprefix));
      //ifstream dump("SOC.Y");
      int N;
      dump >> N;
      if (N != norbs/2) {
        cout << "number of orbitals in SOC.Y should be equal to norbs in the input file."<<endl;
        cout << N <<" != "<<norbs<<endl;
        exit(0);
      }

      //I1soc[2].store.resize(N*(N+1)/2, 0.0);
      while(!dump.eof()) {
        std::getline(dump, msg);
        trim(msg);
        boost::split(tok, msg, is_any_of(", \t="), token_compress_on);
        if (tok.size() != 3) continue;

        double integral = atof(tok[0].c_str());
        int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str());
        I1(2*(a-1), 2*(b-1)+1) += std::complex<double>(integral/2.,0);  //alpha beta
        I1(2*(a-1)+1, 2*(b-1)) += std::complex<double>(-integral/2.,0);  //beta alpha
        //I1(2*(a-1), 2*(b-1)+1) += std::complex<double>(-integral/2.,0);  //alpha beta
        //I1(2*(a-1)+1, 2*(b-1)) += std::complex<double>(integral/2.,0);  //beta alpha
      }
    }

    //Read SOC.Z
    {
      ifstream dump(str(boost::format("%s.Z") % fileprefix));
      //ifstream dump("SOC.Z");
      int N;
      dump >> N;
      if (N != norbs/2) {
        cout << "number of orbitals in SOC.Z should be equal to norbs in the input file."<<endl;
        cout << N <<" != "<<norbs<<endl;
        exit(0);
      }

      //I1soc[3].store.resize(N*(N+1)/2, 0.0);
      while(!dump.eof()) {
        std::getline(dump, msg);
        trim(msg);
        boost::split(tok, msg, is_any_of(", \t="), token_compress_on);
        if (tok.size() != 3) continue;

        double integral = atof(tok[0].c_str());
        int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str());
        I1(2*(a-1), 2*(b-1)) += std::complex<double>(0,integral/2); //alpha, alpha
        I1(2*(a-1)+1, 2*(b-1)+1) += std::complex<double>(0,-integral/2); //beta, beta
        //I1(2*(a-1), 2*(b-1)) += std::complex<double>(0,-integral/2); //alpha, alpha
        //I1(2*(a-1)+1, 2*(b-1)+1) += std::complex<double>(0,integral/2); //beta, beta
      }
    }
  } // commrank=0
} // end readSOCIntegrals
#endif



#ifdef Complex
//=============================================================================
void readGTensorIntegrals(vector<oneInt>& I1, int norbs, string fileprefix) {
//-----------------------------------------------------------------------------
    /*!
    Read GTensor integrals from files, to be put in "I1"

    :Inputs:

        vector<oneInt>& I1:
            One-electron tensor of the Hamiltonian (output)
        int norbs
            Number of orbitals
        string fileprefix
            Basename of the SOC integral files
    */
//-----------------------------------------------------------------------------
  if (commrank == 0) {
    vector<string> tok;
    string msg;

    //Read GTensor.X
    {
      ifstream dump(str(boost::format("%s.X") % fileprefix));
      int N;
      dump >> N;
      if (N != norbs/2) {
        cout << "number of orbitals in SOC.X should be equal to norbs in the input file."<<endl;
        cout << N <<" != "<<norbs<<endl;
        exit(0);
      }

      //I1soc[1].store.resize(N*(N+1)/2, 0.0);
      while(!dump.eof()) {
        std::getline(dump, msg);
        trim(msg);
        boost::split(tok, msg, is_any_of(", \t="), token_compress_on);
        if (tok.size() != 3) continue;

        double integral = atof(tok[0].c_str());
        int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str());
        I1[0](2*(a-1), 2*(b-1)) += std::complex<double>(0, integral);  //alpha alpha
        I1[0](2*(a-1)+1, 2*(b-1)+1) += std::complex<double>(0, integral);  //beta beta
      }
    }

    //Read SOC.Y
    {
      ifstream dump(str(boost::format("%s.Y") % fileprefix));
      //ifstream dump("SOC.Y");
      int N;
      dump >> N;
      if (N != norbs/2) {
        cout << "number of orbitals in SOC.Y should be equal to norbs in the input file."<<endl;
        cout << N <<" != "<<norbs<<endl;
        exit(0);
      }

      //I1soc[2].store.resize(N*(N+1)/2, 0.0);
      while(!dump.eof()) {
        std::getline(dump, msg);
        trim(msg);
        boost::split(tok, msg, is_any_of(", \t="), token_compress_on);
        if (tok.size() != 3) continue;

        double integral = atof(tok[0].c_str());
        int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str());
        I1[1](2*(a-1), 2*(b-1)) += std::complex<double>(0, integral);  //alpha alpha
        I1[1](2*(a-1)+1, 2*(b-1)+1) += std::complex<double>(0, integral);  //beta beta
      }
    }

    //Read SOC.Z
    {
      ifstream dump(str(boost::format("%s.Z") % fileprefix));
      //ifstream dump("SOC.Z");
      int N;
      dump >> N;
      if (N != norbs/2) {
        cout << "number of orbitals in SOC.Z should be equal to norbs in the input file."<<endl;
        cout << N <<" != "<<norbs<<endl;
        exit(0);
      }

      //I1soc[3].store.resize(N*(N+1)/2, 0.0);
      while(!dump.eof()) {
        std::getline(dump, msg);
        trim(msg);
        boost::split(tok, msg, is_any_of(", \t="), token_compress_on);
        if (tok.size() != 3) continue;

        double integral = atof(tok[0].c_str());
        int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str());
        I1[2](2*(a-1), 2*(b-1)) += std::complex<double>(0, integral);  //alpha alpha
        I1[2](2*(a-1)+1, 2*(b-1)+1) += std::complex<double>(0, integral);  //beta beta
      }
    }
  } // commrank=0
} // end readGTensorIntegrals
#endif



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



