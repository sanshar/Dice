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
#include "global.h"
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

using namespace boost;

bool myfn(double i, double j) { return fabs(i)<fabs(j); }



//=============================================================================
void readIntegrals(string fcidump, twoInt& I2, oneInt& I1, 
		   int& nalpha, int& nbeta, int& norbs, double& coreE, 
		   std::vector<int>& irrep) {
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
        int& nalpha:
            Number of alpha electrons (output)
        int& nbeta:
            Number of beta electrons (output)
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
    cout << "Integral file "<<fcidump<<" does not exist!"<<endl;
    exit(0);
  }

  int nelec, sz;

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
    nalpha = nelec/2 + sz;
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
  size_t I2memory = npair*(npair+1)/2; //memory in bytes

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
        I2(2*(a-1),2*(b-1),2*(c-1),2*(d-1)) = integral;
      }
    } // while

    //exit(0);
    I2.maxEntry = *std::max_element(&I2.store[0], &I2.store[0]+I2memory,myfn);
    I2.Direct = MatrixXd::Zero(norbs, norbs); I2.Direct *= 0.;
    I2.Exchange = MatrixXd::Zero(norbs, norbs); I2.Exchange *= 0.;

    for (int i=0; i<norbs; i++)
      for (int j=0; j<norbs; j++) {
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
    MPI::COMM_WORLD.Bcast(&I2.store[i*maxint], maxint, MPI_DOUBLE, 0);
    world.barrier();
  }
  MPI::COMM_WORLD.Bcast(&I2.store[(maxIter)*maxint], I2memory - maxIter*maxint, MPI_DOUBLE, 0);
  world.barrier();

  mpi::broadcast(world, I2.maxEntry, 0);
  mpi::broadcast(world, I2.Direct, 0);
  mpi::broadcast(world, I2.Exchange, 0);
  mpi::broadcast(world, I2.zero, 0);
  mpi::broadcast(world, coreE, 0);
#endif
} // end readIntegrals




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



