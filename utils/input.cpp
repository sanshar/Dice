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
#include "input.h"
#include "CPS.h"
#include "global.h"
#include "Determinants.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>


#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;
using namespace boost;
using namespace std;


void readInput(string input, schedule& schd, bool print) {
  if (commrank == 0)
    {
      if (print)
	{
	  cout << "**************************************************************" << endl;
	  cout << "Input file  :" << endl;
	  cout << "**************************************************************" << endl;
	}

      ifstream dump(input.c_str());

      schd.deterministic = false;
      schd.restart = false;
      schd.expCorrelator = false;

      schd.maxIter = 50;
      schd.avgIter = 0;
      schd._sgdIter = 1;
      schd.method = amsgrad;
      schd.decay2 = 0.001;
      schd.decay1 = 0.1;
      schd.stepsize = 0.001;

      schd.stochasticIter = 1e4;
      schd.integralSampleSize = 10;
      schd.seed = getTime();
      schd.PTlambda = 0.;
      schd.epsilon = 1.e-7;
      schd.screen = 1.e-8;
      schd.determinantFile = "";
      schd.wavefunctionType = "CPSSlater";
      schd.doHessian = false;
      schd.hf = "rhf";
      schd.optimizeOrbs = true;
      schd.optimizeCps = true;
      schd.Hamiltonian = ABINITIO;
      schd.nwalk = 100;
      schd.tau = 0.001;
      schd.fn_factor = 1.0;
      schd.nGeneration = 30.0;
      schd.excitationLevel = 1;
      schd.ctmc = true;
      schd.cgIter = 15;
      schd.sDiagShift = 0.01;

      while (dump.good())
	{

	  std::string
	    Line;
	  std::getline(dump, Line);
	  trim(Line);
	  //if (print)
	  cout << Line << endl;

	  vector<string> tok;
	  boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);
	  string ArgName = *tok.begin();

	  //if (dump.eof())
	  //break;
	  if (!ArgName.empty() && (boost::iequals(tok[0].substr(0, 1), "#")))
	    continue;
	  if (ArgName.empty())
	    continue;

	  if (boost::iequals(ArgName, "restart"))
	    schd.restart = true;

	  else if (boost::iequals(ArgName, "deterministic"))
	    schd.deterministic = true;
	  
          else if (boost::iequals(ArgName, "expCorrelator"))
	    schd.expCorrelator = true;

	  //else if (boost::iequals(ArgName, "adam"))
          //schd.method = adam;

	  else if (boost::iequals(ArgName, "sgd"))
	    schd.method = sgd;

	  //else if (boost::iequals(ArgName, "nestorov"))
          //schd.method = nestorov;

	  //else if (boost::iequals(ArgName, "rmsprop"))
          //schd.method = rmsprop;

	  else if (boost::iequals(ArgName, "ptlambda"))
	    schd.PTlambda = atof(tok[1].c_str());

	  else if (boost::iequals(ArgName, "amsgrad"))
	    schd.method = amsgrad;

	  else if (boost::iequals(ArgName, "sr"))
	    schd.method = sr;

          else if (boost::iequals(ArgName, "sDiagShift"))
            schd.sDiagShift = atof(tok[1].c_str());

          else if (boost::iequals(ArgName, "cgIter"))
            schd.cgIter = atoi(tok[1].c_str());

	  else if (boost::iequals(ArgName, "amsgrad_sgd"))
          {
	    schd.method = amsgrad_sgd;
            schd._sgdIter = atoi(tok[1].c_str());
          }
          
      else if (boost::iequals(ArgName, "jastrowslater"))
	    schd.wavefunctionType = "JastrowSlater";
      
      else if (boost::iequals(ArgName, "cicpsslater"))
	    schd.wavefunctionType = "CICPSSlater";
      
      else if (boost::iequals(ArgName, "cijastrowslater"))
	    schd.wavefunctionType = "CIJastrowSlater";
      
      else if (boost::iequals(ArgName, "lanczoscpsslater"))
	    schd.wavefunctionType = "LanczosCPSSlater";
      
      else if (boost::iequals(ArgName, "lanczosjastrowslater"))
	    schd.wavefunctionType = "LanczosJastrowSlater";
      
      else if (boost::iequals(ArgName, "cpsagp"))
	    schd.wavefunctionType = "CPSAGP";
      
      else if (boost::iequals(ArgName, "jastrowagp"))
	    schd.wavefunctionType = "JastrowAGP";
      
      else if (boost::iequals(ArgName, "cicpsagp"))
	    schd.wavefunctionType = "CICPSAGP";
      
      else if (boost::iequals(ArgName, "cijastrowagp"))
	    schd.wavefunctionType = "CIJastrowAGP";
      
      else if (boost::iequals(ArgName, "lanczoscpsagp"))
	    schd.wavefunctionType = "LanczosCPSAGP";
      
      else if (boost::iequals(ArgName, "lanczosjastrowagp"))
	    schd.wavefunctionType = "LanczosJastrowAGP";
      
      else if (boost::iequals(ArgName, "cpspfaffian"))
	    schd.wavefunctionType = "CPSPfaffian";
      
      else if (boost::iequals(ArgName, "jastrowpfaffian"))
	    schd.wavefunctionType = "JastrowPfaffian";
      
      else if (boost::iequals(ArgName, "cicpspfaffian"))
	    schd.wavefunctionType = "CICPSPfaffian";
      
      else if (boost::iequals(ArgName, "cijastrowpfaffian"))
	    schd.wavefunctionType = "CIJastrowPfaffian";
      
      else if (boost::iequals(ArgName, "lanczoscpspfaffian"))
	    schd.wavefunctionType = "LanczosCPSPfaffian";
      
      else if (boost::iequals(ArgName, "lanczosjastrowpfaffian"))
	    schd.wavefunctionType = "LanczosJastrowPfaffian";
      else if (boost::iequals(ArgName, "ctmc"))
	    schd.ctmc = true;
      else if (boost::iequals(ArgName, "metropolis"))
	    schd.ctmc = false;
	  
	  else if (boost::iequals(ArgName, "tol"))
	    schd.tol = atof(tok[1].c_str());

	  else if (boost::iequals(ArgName, "screentol"))
	    schd.screen = atof(tok[1].c_str());

	  else if (boost::iequals(ArgName, "decay1"))
	    schd.decay1 = atof(tok[1].c_str());

	  else if (boost::iequals(ArgName, "decay2"))
	    schd.decay2 = atof(tok[1].c_str());

	  else if (boost::iequals(ArgName, "epsilon"))
	    schd.epsilon = atof(tok[1].c_str());

	  else if (boost::iequals(ArgName, "seed"))
	    schd.seed = atof(tok[1].c_str());

	  else if (boost::iequals(ArgName, "stepsize"))
	    schd.stepsize = atof(tok[1].c_str());

	  else if (boost::iequals(ArgName, "stochasticiter"))
	    schd.stochasticIter = atoi(tok[1].c_str());

	  else if (boost::iequals(ArgName, "integralsamplesize"))
	    schd.integralSampleSize = atoi(tok[1].c_str());

	  else if (boost::iequals(ArgName, "correlator"))
	    {
	      int siteSize = atoi(tok[1].c_str());
	      schd.correlatorFiles[siteSize] = tok[2];
	    }

	  else if (boost::iequals(ArgName, "determinants"))
	    {
	      schd.determinantFile = tok[1];
	    }


	  else if (boost::iequals(ArgName, "printLevel"))
	    {
	      schd.printLevel = atoi(tok[1].c_str());
	    }

	  else if (boost::iequals(ArgName, "maxiter"))
	    {
	      schd.maxIter = atoi(tok[1].c_str());
	    }
	  
      else if (boost::iequals(ArgName, "avgiter"))
	    {
	      schd.avgIter = atoi(tok[1].c_str());
	    }
	  
	  else if (boost::iequals(ArgName, "doHessian"))
	    {
	      schd.doHessian = true;
	    }
	  
      else if (boost::iequals(ArgName, "rhf"))
	    {
	      schd.hf = "rhf";
	    }

	  else if (boost::iequals(ArgName, "uhf"))
	    {
	      schd.hf = "uhf";
	    }
	  
      else if (boost::iequals(ArgName, "ghf"))
	    {
	      schd.hf = "ghf";
	    }

	  else if (boost::iequals(ArgName, "dontoptimizeorbs"))
	    {
	      schd.optimizeOrbs = false;
	    }
	  
      else if (boost::iequals(ArgName, "dontoptimizecps"))
	    {
	      schd.optimizeCps = false;
	    }

	  else if (boost::iequals(ArgName, "hubbard"))
	    {
	      schd.Hamiltonian = HUBBARD;
	    }

	  else if (boost::iequals(ArgName, "nwalk"))
	    {
	      schd.nwalk = atoi(tok[1].c_str());
	    }

	  else if (boost::iequals(ArgName, "tau"))
	    {
	      schd.tau = atof(tok[1].c_str());
	    }

	  else if (boost::iequals(ArgName, "fixednodefactor"))
	    {
	      schd.fn_factor = atof(tok[1].c_str());
	    }

	  else if (boost::iequals(ArgName, "ngeneration"))
	    {
	      schd.nGeneration = atoi(tok[1].c_str());
	    }

	  else if (boost::iequals(ArgName, "excitationlevel"))
	    {
	      schd.excitationLevel = atoi(tok[1].c_str());
	    }

	  else
	    {
	      cout << "cannot read option " << ArgName << endl;
	      exit(0);
	    }
	}
    }

#ifndef SERIAL
  boost::mpi::communicator world;
  mpi::broadcast(world, schd, 0);
#endif
}

void readCorrelator(const std::pair<int, std::string>& p,
		    std::vector<Correlator>& correlators) {
  readCorrelator(p.second, p.first, correlators);
}

void readCorrelator(std::string input, int correlatorSize,
		    std::vector<Correlator>& correlators) {
  ifstream dump(input.c_str());

  while (dump.good()) {

    std::string
      Line;
    std::getline(dump, Line);
    trim(Line);
    vector<string> tok;
    boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);

    string ArgName = *tok.begin();

    //if (dump.eof())
    //break;
    if (!ArgName.empty() && (boost::iequals(tok[0].substr(0,1), "#"))) continue;
    if (ArgName.empty()) continue;
    
    if (tok.size() != correlatorSize) {
      cout << "Something wrong in line : "<<Line<<endl;
      exit(0);
    }

    vector<int> asites, bsites;
    for (int i=0; i<correlatorSize; i++) {
      int site = atoi(tok[i].c_str());
      asites.push_back(site);
      bsites.push_back(site);
    }
    correlators.push_back(Correlator(asites, bsites));
  }
}


void readHF(MatrixXd& HfmatrixA, MatrixXd& HfmatrixB, std::string hf) 
{
  if (hf == "rhf" || hf == "ghf") {
    ifstream dump("hf.txt");
    for (int i = 0; i < HfmatrixA.rows(); i++) {
      for (int j = 0; j < HfmatrixA.rows(); j++){
        dump >> HfmatrixA(i, j);
	    HfmatrixB(i, j) = HfmatrixA(i, j);
      }
    }
  }
  else {
      ifstream dump("hf.txt");
      for (int i = 0; i < HfmatrixA.rows(); i++)
	{
	  for (int j = 0; j < HfmatrixA.rows(); j++)
	    dump >> HfmatrixA(i, j);
	  for (int j = 0; j < HfmatrixB.rows(); j++)
	    dump >> HfmatrixB(i, j);
	}
    }
/*
  if (schd.optimizeOrbs) {
    double scale = pow(1.*HfmatrixA.rows(), 0.5);
    HfmatrixA += 1.e-2*MatrixXd::Random(HfmatrixA.rows(), HfmatrixA.cols())/scale;
    HfmatrixB += 1.e-2*MatrixXd::Random(HfmatrixB.rows(), HfmatrixB.cols())/scale;
  }
*/
}

void readPairMat(MatrixXd& pairMat) 
{
  ifstream dump("pairMat.txt");
  for (int i = 0; i < pairMat.rows(); i++) {
    for (int j = 0; j < pairMat.rows(); j++){
      dump >> pairMat(i, j);
    }
  }
}

void readDeterminants(std::string input, vector<Determinant> &determinants,
                      vector<double> &ciExpansion)
{
  ifstream dump(input.c_str());
  while (dump.good())
    {
      std::string Line;
      std::getline(dump, Line);

      trim_if(Line, is_any_of(", \t\n"));
      
      vector<string> tok;
      boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);

      if (tok.size() > 2 )
	{
	  ciExpansion.push_back(atof(tok[0].c_str()));
	  determinants.push_back(Determinant());
	  Determinant& det = *determinants.rbegin();
	  for (int i=0; i<Determinant::norbs; i++) 
	    {
	      if (boost::iequals(tok[1+i], "2")) 
		{
		  det.setoccA(i, true);
		  det.setoccB(i, true);
		}
	      else if (boost::iequals(tok[1+i], "a")) 
		{
		  det.setoccA(i, true);
		  det.setoccB(i, false);
		}
	      if (boost::iequals(tok[1+i], "b")) 
		{
		  det.setoccA(i, false);
		  det.setoccB(i, true);
		}
	      if (boost::iequals(tok[1+i], "0")) 
		{
		  det.setoccA(i, false);
		  det.setoccB(i, false);
		}
	    }

	  //***************I AM USING alpha-beta format here, but the wavefunction is coming from Dice that uses alpha0 beta0 alpha1 beta1... format
	  //So the signs need to be adjusted appropriately
	  //cout << det<<"   "<<getParityForDiceToAlphaBeta(det)<<endl;
	  *ciExpansion.rbegin() *= getParityForDiceToAlphaBeta(det);
	}
    }
}
