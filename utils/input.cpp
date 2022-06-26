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
#include <sstream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#ifndef SERIAL
#include "mpi.h"
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif

using namespace Eigen;
using namespace boost;
using namespace std;


void readInput(string inputFile, schedule& schd, bool print) {
  if (commrank == 0) {
    property_tree::iptree input;
    property_tree::read_json(inputFile, input);
    
    //print input file
    if (print)	{
      cout << "**************************************************************" << endl;
      cout << "Input file  :" << endl;
      cout << "**************************************************************" << endl;
      stringstream ss;
      property_tree::json_parser::write_json(ss, input);
      cout << ss.str() << endl;
    }

    // system options
    schd.integralsFile = input.get("system.integrals", "FCIDUMP");
    schd.intType = input.get("system.intType", "r");
    schd.nciCore = input.get("system.numCore", 0);                  // TODO: rename these because active spaces are also used without ci
    schd.nciAct = input.get("system.numAct", -1);
    schd.soc = input.get("system.soc", false);
    
    //minimal checking for correctness
    //wavefunction
    schd.wavefunctionType = algorithm::to_lower_copy(input.get("wavefunction.name", "jastrowslater"));
    
    //correlatedwavefunction options
    schd.hf = algorithm::to_lower_copy(input.get("wavefunction.hfType", "rhf"));
    schd.ifComplex = input.get("wavefunction.complex", false);
    schd.uagp = input.get("wavefunction.uagp", false); 
    optional< property_tree::iptree& > child = input.get_child_optional("wavefunction.correlators");
    if (child) {
      for (property_tree::iptree::value_type &correlator : input.get_child("wavefunction.correlators")) {
        int siteSize = stoi(correlator.first);
        string file = correlator.second.data();
	    schd.correlatorFiles[siteSize] = file;
      }
    }
    
    //jastrow multislater
    schd.ghfDets = input.get("wavefunction.ghfDets", false);
    schd.ciThreshold = input.get("wavefunction.ciThreshold", 1.e-5);

    //resonating wave function
    schd.numResonants = input.get("wavefunction.numResonants", 1);
    schd.singleJastrow = input.get("wavefunction.singleJastrow", true);
    schd.readTransOrbs = input.get("wavefunction.readTransOrbs", true);
    
    // permuted wave function
    schd.numPermutations = input.get("wavefunction.numPermutations", 1);

    //ci and lanczos
    schd.usingFOIS = false;
    schd.overlapCutoff = input.get("wavefunction.overlapCutoff", 1.e-5);
    if (schd.wavefunctionType == "sci") schd.ciCeption = true;
    else schd.ciCeption = false;
    schd.determinantFile = input.get("wavefunction.determinants", ""); //used for both sci and starting det
    schd.detsInCAS = input.get("wavefunction.detsInCAS", true);
    schd.alpha = input.get("wavefunction.alpha", 0.01); //lanczos
    schd.lanczosEpsilon = input.get("wavefunction.lanczosEpsilon", 1.e-8); //lanczos

    // nnb and rbm
    schd.numHidden = input.get("wavefunction.numHidden", 1);
    schd.numHiddenLayers = input.get("wavefunction.numHiddenLayers", 1);

    // multi-Slater
    schd.excitationLevel = input.get("wavefunction.excitationLevel", 10);

    //hamiltonian
    string hamString = algorithm::to_lower_copy(input.get("hamiltonian", "abinitio"));
    if (hamString == "abinitio") schd.Hamiltonian = ABINITIO;
    else if (hamString == "hubbard") schd.Hamiltonian = HUBBARD;
    
    //sampling
    schd.epsilon = input.get("sampling.epsilon", 1.e-7);
    schd.screen = input.get("sampling.screentol", 1.e-8);
    schd.ctmc = input.get("sampling.ctmc", true); //if this is false, metropolis is used!
    schd.deterministic = input.get("sampling.deterministic", false);
    schd.stochasticIter = input.get("sampling.stochasticIter", 1e4);
    schd.burnIter = input.get("sampling.burnIter", 0);
    schd.integralSampleSize = input.get("sampling.integralSampleSize", 10);
    schd.useLastDet = input.get("sampling.useLastDet", false);
    schd.useLogTime = input.get("sampling.useLogTime", false);
    schd.numSCSamples = input.get("sampling.numSCSamples", 1e3);
    schd.normSampleThreshold = input.get("sampling.normSampleThreshold", 5.);
    schd.seed = input.get("sampling.seed", getTime());

    // SC-NEVPT2(s) sampling options:
    schd.determCCVV = input.get("sampling.determCCVV", false);
    schd.efficientNEVPT = input.get("sampling.efficientNEVPT", false);
    schd.efficientNEVPT_2 = input.get("sampling.efficientNEVPT_2", false);
    schd.exactE_NEVPT = input.get("sampling.exactE_NEVPT", false);
    schd.NEVPT_readE = input.get("sampling.NEVPT_readE", false);
    schd.NEVPT_writeE = input.get("sampling.NEVPT_writeE", false);
    schd.continueMarkovSCPT = input.get("sampling.continueMarkovSCPT", true);
    schd.stochasticIterNorms = input.get("sampling.stochasticIterNorms", 1e4);
    schd.stochasticIterEachSC = input.get("sampling.stochasticIterEachSC", 1e2);
    schd.nIterFindInitDets = input.get("sampling.nIterFindInitDets", 1e2);
    schd.SCEnergiesBurnIn = input.get("sampling.SCEnergiesBurnIn", 50);
    schd.SCNormsBurnIn = input.get("sampling.SCNormsBurnIn", 50);
    schd.exactPerturber = input.get("sampling.exactPerturber", false);
    schd.perturberOrb1 = input.get("sampling.perturberOrb1", -1);
    schd.perturberOrb2 = input.get("sampling.perturberOrb2", -1);
    schd.fixedResTimeNEVPT_Ene = input.get("sampling.fixedResTimeNEVPT_Ene", true);
    schd.fixedResTimeNEVPT_Norm = input.get("sampling.fixedResTimeNEVPT_Norm", false);
    schd.resTimeNEVPT_Ene = input.get("sampling.resTimeNEVPT_Ene", 5.0);
    schd.resTimeNEVPT_Norm = input.get("sampling.resTimeNEVPT_Norm", 5.0);
    schd.CASEnergy = input.get("sampling.CASEnergy", 0.0);

    // dqmc
    schd.nsteps = input.get("sampling.nsteps", 10);
    child = input.get_child_optional("sampling.eneSteps");
    if (child) {
      for (property_tree::iptree::value_type &eneSteps : input.get_child("sampling.eneSteps")) {
        schd.eneSteps.push_back(stoi(eneSteps.second.data()) - 1);
      }
    }
    else {
      schd.eneSteps =  { int(0.4*schd.nsteps) - 1, int(0.6*schd.nsteps) - 1, int(0.8*schd.nsteps) - 1, int(schd.nsteps - 1) };
    }
    child = input.get_child_optional("sampling.errorTargets");
    if (child) {
      for (property_tree::iptree::value_type &errorTargets : input.get_child("sampling.errorTargets")) {
        schd.errorTargets.push_back(stof(errorTargets.second.data()));
      }
    }
    else {
      for (int i = 0; i < schd.eneSteps.size(); i++) schd.errorTargets.push_back(1.5e-3);
    }
    schd.printFrequency = input.get("sampling.printFreq", 100);
    schd.dt = input.get("sampling.dt", 0.1);
    schd.fieldStepsize = input.get("sampling.stepsize", 0.1);
    schd.measureFreq = input.get("sampling.measureFreq", 10);
    schd.orthoSteps = input.get("sampling.orthoSteps", 50);
    schd.ene0Guess = input.get("sampling.ene0Guess", 1e10); // assuming ground state energy will not be 1e10
    schd.numJastrowSamples = input.get("sampling.numJastrowSamples", 50);
    schd.ngrid = input.get("sampling.ngrid", 1);
    schd.sampleDeterminants = input.get("sampling.sampleDeterminants", -1);
    schd.choleskyThreshold = input.get("sampling.choleskyThreshold", 0.005);
    schd.leftWave = algorithm::to_lower_copy(input.get("wavefunction.left", "rhf"));
    schd.rightWave = algorithm::to_lower_copy(input.get("wavefunction.right", "rhf"));
    schd.ndets = input.get("wavefunction.ndets", 1e6);
    schd.phaseless = input.get("sampling.phaseless", false);
    schd.weightCap = input.get("sampling.weightCap", -1.);

    // GFMC
    schd.maxIter = input.get("sampling.maxIter", 50); //note: parameter repeated in optimizer for vmc
    schd.nwalk = input.get("sampling.nwalk", 100);
    schd.tau = input.get("sampling.tau", 0.001);
    schd.fn_factor = input.get("sampling.fn_factor", 1.0);
    schd.nGeneration = input.get("sampling.nGeneration", 30.0);
    
    // FCIQMC options
    schd.maxIterFCIQMC = input.get("FCIQMC.maxIter", 50);
    schd.nreplicas = input.get("FCIQMC.nReplicas", 1);
    schd.nAttemptsEach = input.get("FCIQMC.nAttemptsEach", 1);
    schd.mainMemoryFac = input.get("FCIQMC.mainMemoryFac", 5.0);
    schd.spawnMemoryFac = input.get("FCIQMC.spawnMemoryFac", 5.0);
    schd.shiftDamping = input.get("FCIQMC.shiftDamping", 0.01);
    schd.initialShift = input.get("FCIQMC.initialShift", 0.0);
    schd.minSpawn = input.get("FCIQMC.minSpawn", 0.01);
    schd.minPop = input.get("FCIQMC.minPop", 1.0);
    schd.initialPop = input.get("FCIQMC.initialPop", 100.0);
    schd.initialNDets = input.get("FCIQMC.initialNDets", 1);
    schd.trialInitFCIQMC = input.get("FCIQMC.trialInit", false);
    schd.targetPop = input.get("FCIQMC.targetPop", 1000.0);
    schd.initiator = input.get("FCIQMC.initiator", false);
    schd.initiatorThresh = input.get("FCIQMC.initiatorThresh", 2.0);
    schd.semiStoch = input.get("FCIQMC.semiStoch", false);
    schd.semiStochInit = input.get("FCIQMC.semiStochInit", false);
    schd.semiStochFile = input.get("wavefunction.semiStochDets", "dets");
    schd.uniformExGen = input.get("FCIQMC.uniform", true);
    schd.heatBathExGen = input.get("FCIQMC.heatBath", false);
    schd.heatBathUniformSingExGen = input.get("FCIQMC.heatBathUniformSingles", false);
    schd.calcEN2 = input.get("FCIQMC.EN2", false);
    schd.useTrialFCIQMC = input.get("FCIQMC.useTrial", false);
    schd.trialWFEstimator = input.get("FCIQMC.trialWFEstimator", false);
    schd.importanceSampling = input.get("FCIQMC.importanceSampling", false);
    schd.applyNodeFCIQMC = input.get("FCIQMC.applyNode", false);
    schd.releaseNodeFCIQMC = input.get("FCIQMC.releaseNode", false);
    schd.releaseNodeIter = input.get("FCIQMC.releaseNodeIter", 2000);
    schd.diagonalDumping = input.get("FCIQMC.diagonalDumping", false);
    schd.partialNodeFactor = input.get("FCIQMC.partialNodeFactor", 1.0);
    schd.expApprox = input.get("FCIQMC.expApprox", false);
    schd.printAnnihilStats = input.get("FCIQMC.printAnnihilStats", true);

    //optimization
    string method = algorithm::to_lower_copy(input.get("optimizer.method", "amsgrad")); 
    //need a better way of doing this
    if (method == "amsgrad") schd.method = amsgrad;
    else if (method == "amsgrad_sgd") schd.method = amsgrad_sgd;
    else if (method == "sgd") schd.method = sgd;
    else if (method == "sr") schd.method = sr;
    else if (method == "lm") schd.method = linearmethod;
    schd.restart = input.get("optimizer.restart", false);
    schd.fullRestart = input.get("optimizer.fullRestart", false);
    child = input.get_child_optional("sampling.maxIter"); //to ensure maxiter is not reassigned
    if (!child) schd.maxIter = input.get("optimizer.maxIter", 50);
    schd.avgIter = input.get("optimizer.avgIter", 0);
    schd._sgdIter = input.get("optimizer.sgdIter", 1);
    schd.decay2 = input.get("optimizer.decay2", 0.001);
    schd.decay1 = input.get("optimizer.decay1", 0.1);
    schd.momentum = input.get("optimizer.momentum", 0.);
    schd.stepsize = input.get("optimizer.stepsize", 0.001);
    schd.optimizeOrbs = input.get("optimizer.optimizeOrbs", true);
    schd.optimizeCiCoeffs = input.get("optimizer.optimizeCiCoeffs", true);
    schd.optimizeCps = input.get("optimizer.optimizeCps", true); // this is used for all correlators in correlatedwavefunction, not just cps
    schd.optimizeJastrow = input.get("optimizer.optimizeJastrow", true); // this is misleading, becuase this is only relevant to jrbm
    schd.optimizeRBM = input.get("optimizer.optimizeRBM", true);
    schd.cgIter = input.get("optimizer.cgIter", 15);
    schd.sDiagShift = input.get("optimizer.sDiagShift", 0.01);
    schd.doHessian = input.get("optimizer.doHessian", false);
    schd.diagMethod = input.get("optimizer.diagMethod", "power");
    schd.powerShift = input.get("optimizer.powerShift", 10);

    //debug and print options
    schd.printLevel = input.get("print.level", 0);
    schd.printVars = input.get("print.vars", false);
    schd.printGrad = input.get("print.grad", false);
    schd.printJastrow = input.get("print.jastrow", false);
    schd.debug = input.get("print.debug", false);
    // SC-NEVPT(2) print options:
    schd.printSCNorms = input.get("print.SCNorms", true);
    schd.printSCNormFreq = input.get("print.SCNormFreq", 1);
    schd.readSCNorms = input.get("print.readSCNorms", false);
    schd.continueSCNorms = input.get("print.continueSCNorms", false);
    schd.sampleNEVPT2Energy = input.get("print.sampleNEVPT2Energy", true);
    schd.printSCEnergies = input.get("print.SCEnergies", false);
    schd.nWalkSCEnergies = input.get("print.nWalkSCEnergies", 1);
    // dqmc
    schd.writeOneRDM = input.get("print.writeOneRDM", false);
    schd.scratchDir = input.get("print.scratchDir", "./");


    //deprecated, or I don't know what they do
    schd.actWidth = input.get("wavefunction.actWidth", 100);
    schd.numActive = input.get("wavefunction.numActive", -1);
    schd.expCorrelator = input.get("wavefunction.expCorrelator", false); 
    schd.PTlambda = input.get("PTlambda", 0.);
    schd.tol = input.get("tol", 0.); 
    schd.beta = input.get("beta", 1.);
    
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

void readMat(MatrixXd& mat, std::string fileName) 
{
  ifstream dump(fileName);
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++){
      dump >> mat(i, j);
    }
  }
}

void readMat(MatrixXd& mat, hid_t& file, std::string datasetName) {
  int nrows = mat.rows(), ncols = mat.cols();
  double *matRead = new double[nrows * ncols];
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      matRead[i * ncols + j] = 0.;
    }
  }

  hid_t dataset = (-1);
  herr_t status;
  H5E_BEGIN_TRY {
    dataset = H5Dopen(file, datasetName.c_str(), H5P_DEFAULT);
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, matRead);
  } H5E_END_TRY
  if (dataset < 0) {
    if (commrank == 0) cout << datasetName << " dataset could not be read." << endl;
    exit(1);
  }
  
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      mat(i, j) = matRead[i * ncols + j];
    }
  }
  delete [] matRead;
}

void readMat(MatrixXcd& mat, hid_t& file, std::string datasetName) {
  int nrows = mat.rows(), ncols = mat.cols();
  MatrixXd matR = Eigen::MatrixXd::Zero(nrows, ncols);
  MatrixXd matI = Eigen::MatrixXd::Zero(nrows, ncols);
  std::string datasetRName = datasetName + "_real";
  std::string datasetIName = datasetName + "_imag";
  readMat(matR, file, datasetRName);
  readMat(matI, file, datasetIName);
  mat = matR + std::complex<double>(0., 1.) * matI;
}

void writeMat(MatrixXcd& mat, std::string fileName) {
  ofstream dump(fileName);
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++){
      dump << mat(i, j);
    }
  }
}

void readMat(MatrixXcd& mat, std::string fileName) 
{
  ifstream dump(fileName);
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++){
      dump >> mat(i, j);
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

void readDeterminants(std::string input, std::vector<int>& ref, std::vector<int>& open, std::vector<std::array<VectorXi, 2>>& ciExcitations,
        std::vector<int>& ciParity, std::vector<double>& ciCoeffs)
{
  int norbs = Determinant::norbs;
  ifstream dump(input.c_str());
  bool isFirst = true;
  Determinant refDet;
  VectorXi sizes = VectorXi::Zero(10);
  int numDets = 0;
  
  while (dump.good()) {
    std::string Line;
    std::getline(dump, Line);

    boost::trim_if(Line, boost::is_any_of(", \t\n"));
    
    vector<string> tok;
    boost::split(tok, Line, boost::is_any_of(", \t\n"), boost::token_compress_on);


    if (tok.size() > 2 ) {
      if (isFirst) {//first det is ref
        isFirst = false;
        ciCoeffs.push_back(atof(tok[0].c_str()));
        ciParity.push_back(1);
        std::array<VectorXi, 2> empty;
        ciExcitations.push_back(empty);
        vector<int> closedBeta, openBeta; //no ghf det structure, so artificially using vector of ints, alpha followed by beta
        for (int i=0; i<norbs; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            refDet.setoccA(i, true);
            refDet.setoccB(i, true);
            ref.push_back(i);
            closedBeta.push_back(i + norbs);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            refDet.setoccA(i, true);
            ref.push_back(i);
            openBeta.push_back(i + norbs);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            refDet.setoccB(i, true);
            closedBeta.push_back(i + norbs);
            open.push_back(i);
          }
          else if (boost::iequals(tok[1+i], "0")) {
            open.push_back(i);
            openBeta.push_back(i + norbs);
          }
        }
        ref.insert(ref.end(), closedBeta.begin(), closedBeta.end());
        open.insert(open.end(), openBeta.begin(), openBeta.end());
      }
      else {
        vector<int> desA, creA, desB, creB;
        for (int i=0; i<norbs; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            if (!refDet.getoccA(i)) creA.push_back(i);
            if (!refDet.getoccB(i)) creB.push_back(i);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            if (!refDet.getoccA(i)) creA.push_back(i);
            if (refDet.getoccB(i)) desB.push_back(i);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            if (refDet.getoccA(i)) desA.push_back(i);
            if (!refDet.getoccB(i)) creB.push_back(i);
          }
          else if (boost::iequals(tok[1+i], "0")) {
            if (refDet.getoccA(i)) desA.push_back(i);
            if (refDet.getoccB(i)) desB.push_back(i);
          }
        }
        VectorXi cre = VectorXi::Zero(creA.size() + creB.size());
        VectorXi des = VectorXi::Zero(desA.size() + desB.size());
        for (int i = 0; i < creA.size(); i++) {
          des[i] = std::search_n(ref.begin(), ref.end(), 1, desA[i]) - ref.begin();
          cre[i] = std::search_n(open.begin(), open.end(), 1, creA[i]) - open.begin();
          //cre[i] = creA[i];
        }
        for (int i = 0; i < creB.size(); i++) {
          des[i + desA.size()] = std::search_n(ref.begin(), ref.end(), 1, desB[i] + norbs) - ref.begin();
          cre[i + creA.size()] = std::search_n(open.begin(), open.end(), 1, creB[i] + norbs) - open.begin();
          //cre[i + creA.size()] = creB[i] + norbs;
        }
        std::array<VectorXi, 2> excitations;
        excitations[0] = des;
        excitations[1] = cre;
        if (cre.size() > schd.excitationLevel) continue;
        numDets++;
        ciCoeffs.push_back(atof(tok[0].c_str()));
        ciParity.push_back(refDet.parityA(creA, desA) * refDet.parityB(creB, desB));
        ciExcitations.push_back(excitations);
        if (cre.size() < 10) sizes(cre.size())++;
      }
    }
  }
  if (commrank == 0) {
    cout << "Rankwise number of excitations " << sizes.transpose() << endl;
    cout << "Number of determinants " << numDets << endl << endl;
  }
}

void readDeterminantsGHF(std::string input, std::vector<int>& ref, std::vector<int>& open, std::vector<std::array<VectorXi, 2>>& ciExcitations,
        std::vector<int>& ciParity, std::vector<double>& ciCoeffs)
{
  int norbs = Determinant::norbs;
  ifstream dump(input.c_str());
  bool isFirst = true;
  Determinant refDet;

  while (dump.good()) {
    std::string Line;
    std::getline(dump, Line);

    boost::trim_if(Line, boost::is_any_of(", \t\n"));
    
    vector<string> tok;
    boost::split(tok, Line, boost::is_any_of(", \t\n"), boost::token_compress_on);

    if (tok.size() > 2 ) {
      if (isFirst) {//first det is ref
        isFirst = false;
        ciCoeffs.push_back(atof(tok[0].c_str()));
        ciParity.push_back(1);
        std::array<VectorXi, 2> empty;
        ciExcitations.push_back(empty);
        for (int i=0; i<norbs; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            refDet.setoccA(2*i, true);
            refDet.setoccA(2*i+1, true);
            ref.push_back(2*i);
            ref.push_back(2*i+1);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            refDet.setoccA(2*i, true);
            ref.push_back(2*i);
            open.push_back(2*i+1);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            refDet.setoccA(2*i+1, true);
            ref.push_back(2*i+1);
            open.push_back(2*i);
          }
          else if (boost::iequals(tok[1+i], "0")) {
            open.push_back(2*i);
            open.push_back(2*i+1);
          }
        }
      }
      else {
        ciCoeffs.push_back(atof(tok[0].c_str()));
        vector<int> des, cre;
        for (int i=0; i<norbs; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            if (!refDet.getoccA(2*i)) cre.push_back(2*i);
            if (!refDet.getoccA(2*i+1)) cre.push_back(2*i+1);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            if (!refDet.getoccA(2*i)) cre.push_back(2*i);
            if (refDet.getoccA(2*i+1)) des.push_back(2*i+1);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            if (refDet.getoccA(2*i)) des.push_back(2*i);
            if (!refDet.getoccA(2*i+1)) cre.push_back(2*i+1);
          }
          else if (boost::iequals(tok[1+i], "0")) {
            if (refDet.getoccA(2*i)) des.push_back(2*i);
            if (refDet.getoccA(2*i+1)) des.push_back(2*i+1);
          }
        }
        ciParity.push_back(refDet.parityA(cre, des));
        VectorXi creV = VectorXi::Zero(cre.size());
        VectorXi desV = VectorXi::Zero(des.size());
        for (int i = 0; i < cre.size(); i++) {
          desV[i] = std::search_n(ref.begin(), ref.end(), 1, des[i]) - ref.begin();
          creV[i] = std::search_n(open.begin(), open.end(), 1, cre[i]) - open.begin();
          //creV[i] = cre[i];
        }
        std::array<VectorXi, 2> excitations;
        excitations[0] = desV;
        excitations[1] = creV;
        ciExcitations.push_back(excitations);
      }
    }
  }
}


void readDeterminants(std::string input, std::array<std::vector<int>, 2>& ref, std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2>& ciExcitations,
        std::vector<double>& ciParity, std::vector<double>& ciCoeffs)
{
  int norbs = Determinant::norbs;
  int nact = norbs;
  if (schd.nciAct > 0) nact = schd.nciAct;
  int ncore = 0;
  if (schd.nciCore > 0) ncore = schd.nciCore;
  ifstream dump(input.c_str());
  bool isFirst = true;
  Determinant refDet;
  for (int i = 0; i < ncore; i++) {
    refDet.setoccA(i, true);
    refDet.setoccB(i, true);
    ref[0].push_back(i);
    ref[1].push_back(i);
  }
  VectorXi sizes = VectorXi::Zero(10);
  int numDets = 0;
  
  while (dump.good()) {
    std::string Line;
    std::getline(dump, Line);

    boost::trim_if(Line, boost::is_any_of(", \t\n"));
    
    vector<string> tok;
    boost::split(tok, Line, boost::is_any_of(", \t\n"), boost::token_compress_on);

    if (tok.size() > 2 ) {
      if (isFirst) {// first det is ref
        isFirst = false;
        ciCoeffs.push_back(atof(tok[0].c_str()));
        ciParity.push_back(1);
        std::array<VectorXi, 2> empty;
        ciExcitations[0].push_back(empty);
        ciExcitations[1].push_back(empty);
        vector<int> closedBeta, openBeta; 
        for (int i = 0; i < nact; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            refDet.setoccA(ncore + i, true);
            refDet.setoccB(ncore + i, true);
            ref[0].push_back(ncore + i);
            ref[1].push_back(ncore + i);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            refDet.setoccA(ncore + i, true);
            ref[0].push_back(ncore + i);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            refDet.setoccB(ncore + i, true);
            ref[1].push_back(ncore + i);
          }
        }
        numDets++;
        sizes(0) = 1;
      }
      else {
        vector<int> desA, creA, desB, creB;
        for (int i = 0; i < nact; i++) {
          if (boost::iequals(tok[1+i], "2")) {
            if (!refDet.getoccA(ncore + i)) creA.push_back(ncore + i);
            if (!refDet.getoccB(ncore + i)) creB.push_back(ncore + i);
          }
          else if (boost::iequals(tok[1+i], "a")) {
            if (!refDet.getoccA(ncore + i)) creA.push_back(ncore + i);
            if (refDet.getoccB(ncore + i)) desB.push_back(ncore + i);
          }
          else if (boost::iequals(tok[1+i], "b")) {
            if (refDet.getoccA(ncore + i)) desA.push_back(ncore + i);
            if (!refDet.getoccB(ncore + i)) creB.push_back(ncore + i);
          }
          else if (boost::iequals(tok[1+i], "0")) {
            if (refDet.getoccA(ncore + i)) desA.push_back(ncore + i);
            if (refDet.getoccB(ncore + i)) desB.push_back(ncore + i);
          }
        }

        std::array<VectorXi, 2> excitationsA;
        excitationsA[0] = VectorXi::Zero(creA.size());
        excitationsA[1] = VectorXi::Zero(desA.size());
        for (int i = 0; i < creA.size(); i++) {
          //des[i] = std::search_n(ref.begin(), ref.end(), 1, desA[i]) - ref.begin();
          //cre[i] = std::search_n(open.begin(), open.end(), 1, creA[i]) - open.begin();
          excitationsA[0](i) = desA[i];
          excitationsA[1](i) = creA[i];
        }

        std::array<VectorXi, 2> excitationsB;
        excitationsB[0] = VectorXi::Zero(creB.size());
        excitationsB[1] = VectorXi::Zero(desB.size());
        for (int i = 0; i < creB.size(); i++) {
          //des[i + desA.size()] = std::search_n(ref.begin(), ref.end(), 1, desB[i] + norbs) - ref.begin();
          //cre[i + creA.size()] = std::search_n(open.begin(), open.end(), 1, creB[i] + norbs) - open.begin();
          excitationsB[0](i) = desB[i];
          excitationsB[1](i) = creB[i];
        }
        
        if (creA.size() + creB.size() > schd.excitationLevel) continue;
        numDets++;
        ciCoeffs.push_back(atof(tok[0].c_str()));
        ciParity.push_back(refDet.parityA(creA, desA) * refDet.parityB(creB, desB));
        ciExcitations[0].push_back(excitationsA);
        ciExcitations[1].push_back(excitationsB);
        if (creA.size() + creB.size() < 10) sizes(creA.size() + creB.size())++;
      }
    }
  }
  if (commrank == 0) {
    cout << "Rankwise number of excitations " << sizes.transpose() << endl;
    cout << "Number of determinants " << numDets << endl << endl;
  }
}

// same as above but for binary files
void readDeterminantsBinary(std::string input, std::array<std::vector<int>, 2>& ref, std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2>& ciExcitations,
        std::vector<double>& ciParity, std::vector<double>& ciCoeffs)
{
  int norbs = Determinant::norbs;
  int nact = norbs;
  if (schd.nciAct > 0) nact = schd.nciAct;
  int ncore = 0;
  if (schd.nciCore > 0) ncore = schd.nciCore;
  int ndetsDice = 0, norbsDice = 0;
  ifstream dump(input, ios::binary);
  dump.read((char*) &ndetsDice, sizeof(int));
  dump.read((char*) &norbsDice, sizeof(int));
  bool isFirst = true;
  Determinant refDet;
  for (int i = 0; i < ncore; i++) {
    refDet.setoccA(i, true);
    refDet.setoccB(i, true);
    ref[0].push_back(i);
    ref[1].push_back(i);
  }
  VectorXi sizes = VectorXi::Zero(10);
  int numDets = 0;
  std::array<std::vector<int>, 2> open;

  for (int n = 0; n < ndetsDice; n++) {
    if (isFirst) {// first det is ref
      isFirst = false;
      double ciCoeff;
      dump.read((char*) &ciCoeff, sizeof(double));
      ciCoeffs.push_back(ciCoeff);
      ciParity.push_back(1);
      std::array<VectorXi, 2> empty;
      ciExcitations[0].push_back(empty);
      ciExcitations[1].push_back(empty);
      vector<int> closedBeta, openBeta; 
      for (int i = ncore; i < ncore + norbsDice; i++) {
        char detocc;
        dump.read((char*) &detocc, sizeof(char));
        if (detocc == '2') {
          refDet.setoccA(i, true);
          refDet.setoccB(i, true);
          ref[0].push_back(i);
          ref[1].push_back(i);
        }
        else if (detocc == 'a') {
          refDet.setoccA(i, true);
          ref[0].push_back(i);
          open[1].push_back(i);
        }
        else if (detocc == 'b') {
          refDet.setoccB(i, true);
          ref[1].push_back(i);
          open[0].push_back(i);
        }
        else if (detocc == '0') {
          open[0].push_back(i);
          open[1].push_back(i);
        }
      }
      numDets++;
      sizes(0) = 1;
    }
    else {
      double ciCoeff;
      dump.read((char*) &ciCoeff, sizeof(double));
      vector<int> desA, creA, desB, creB;
      for (int i = ncore; i < ncore + norbsDice; i++) {
        char detocc;
        dump.read((char*) &detocc, sizeof(char));
        if (detocc == '2') {
          if (!refDet.getoccA(i)) creA.push_back(i);
          if (!refDet.getoccB(i)) creB.push_back(i);
        }
        else if (detocc == 'a') {
          if (!refDet.getoccA(i)) creA.push_back(i);
          if (refDet.getoccB(i)) desB.push_back(i);
        }
        else if (detocc == 'b') {
          if (refDet.getoccA(i)) desA.push_back(i);
          if (!refDet.getoccB(i)) creB.push_back(i);
        }
        else if (detocc == '0') {
          if (refDet.getoccA(i)) desA.push_back(i);
          if (refDet.getoccB(i)) desB.push_back(i);
        }
      }

      std::array<VectorXi, 2> excitationsA;
      excitationsA[0] = VectorXi::Zero(creA.size());
      excitationsA[1] = VectorXi::Zero(desA.size());
      for (int i = 0; i < creA.size(); i++) {
        //des[i] = std::search_n(ref.begin(), ref.end(), 1, desA[i]) - ref.begin();
        //cre[i] = std::search_n(open.begin(), open.end(), 1, creA[i]) - open.begin();
        excitationsA[0](i) = std::search_n(ref[0].begin(), ref[0].end(), 1, desA[i]) - ref[0].begin();
        //excitationsA[1](i) = std::search_n(open[0].begin(), open[0].end(), 1, creA[i]) - open[0].begin();
        //excitationsA[0](i) = desA[i];
        excitationsA[1](i) = creA[i];
      }

      std::array<VectorXi, 2> excitationsB;
      excitationsB[0] = VectorXi::Zero(creB.size());
      excitationsB[1] = VectorXi::Zero(desB.size());
      for (int i = 0; i < creB.size(); i++) {
        //des[i + desA.size()] = std::search_n(ref.begin(), ref.end(), 1, desB[i] + norbs) - ref.begin();
        //cre[i + creA.size()] = std::search_n(open.begin(), open.end(), 1, creB[i] + norbs) - open.begin();
        excitationsB[0](i) = std::search_n(ref[1].begin(), ref[1].end(), 1, desB[i]) - ref[1].begin();
        //excitationsB[1](i) = std::search_n(open[1].begin(), open[1].end(), 1, creB[i]) - open[1].begin();
        //excitationsB[0](i) = desB[i];
        excitationsB[1](i) = creB[i];
      }
      
      if (creA.size() + creB.size() > schd.excitationLevel) continue;
      if (abs(ciCoeff) < schd.ciThreshold || numDets == schd.ndets) break;
      numDets++;
      ciCoeffs.push_back(ciCoeff);
      ciParity.push_back(refDet.parityA(creA, desA) * refDet.parityB(creB, desB));
      ciExcitations[0].push_back(excitationsA);
      ciExcitations[1].push_back(excitationsB);
      if (creA.size() + creB.size() < 10) sizes(creA.size() + creB.size())++;
    }
  }
  if (commrank == 0) {
    ofstream afqmcFile("afqmc.dat", ios::app);
    afqmcFile << "# Rankwise number of excitations " << sizes.transpose() << endl;
    afqmcFile << "# Number of determinants " << numDets << "\n#\n";
    afqmcFile.close();
  }
}

void readSpinRDM(std::string fname, Eigen::MatrixXd& oneRDM, Eigen::MatrixXd& twoRDM) {
  // Read a 2-RDM from the spin-RDM text file output by Dice
  // Also construct the 1-RDM at the same time

  int nSpinOrbs = 2*Determinant::norbs;
  int nPairs = nSpinOrbs * nSpinOrbs;

  oneRDM = MatrixXd::Zero(nSpinOrbs, nSpinOrbs);
  twoRDM = MatrixXd::Zero(nPairs, nPairs);

  ifstream RDMFile(fname);
  string lineStr;
  while (getline(RDMFile, lineStr)) {
    string buf;
    stringstream ss(lineStr);
    vector<string> words;
    while (ss >> buf) words.push_back(buf);

    int a = stoi(words[0]);
    int b = stoi(words[1]);
    int c = stoi(words[2]);
    int d = stoi(words[3]);
    double elem = stod(words[4]);

    int ind1 = a * nSpinOrbs + b;
    int ind2 = c * nSpinOrbs + d;
    int ind3 = b * nSpinOrbs + a;
    int ind4 = d * nSpinOrbs + c;

    twoRDM(ind1, ind2) = elem;
    twoRDM(ind3, ind2) = -elem;
    twoRDM(ind1, ind4) = -elem;
    twoRDM(ind3, ind4) = elem;

    if (b == d) oneRDM(a,c) += elem;
    if (b == c) oneRDM(a,d) += -elem;
    if (a == d) oneRDM(b,c) += -elem;
    if (a == c) oneRDM(b,d) += elem;
  }

  int nelec = Determinant::nalpha + Determinant::nbeta;

  // Normalize the 1-RDM
  for (int a = 0; a < nSpinOrbs; a++) {
    for (int b = 0; b < nSpinOrbs; b++) {
      oneRDM(a,b) /= nelec-1;
    }
  }
}


// reads ccsd amplitudes
void readCCSD(Eigen::MatrixXd& singles, Eigen::MatrixXd& doubles, Eigen::MatrixXd& basisRotation, string fname) 
{
  int nocc = singles.rows(), nopen = singles.cols();
  int norbs = nocc + nopen;
  int nexc = nocc * nopen;

  hid_t file = (-1), dataset_singles, dataset_doubles, dataset_rotation;  /* identifiers */
  herr_t status;

  H5E_BEGIN_TRY {
    file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  } H5E_END_TRY
 
  if (file < 0) {
    if (commrank == 0) cout << "Amplitudes not found!" << endl;
    exit(0);
  }
  
  double *singlesMem = new double[ nocc * nopen ];
  for (int i = 0; i < nocc; i++) 
    for (int j = 0; j < nopen; j++)
      singlesMem[i * nopen + j] = 0.;
  dataset_singles = H5Dopen(file, "/singles", H5P_DEFAULT);
  status = H5Dread(dataset_singles, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, singlesMem);
  for (int i = 0; i < nocc; i++) {
    for (int j = 0; j < nopen; j++) {
      singles(i, j) = singlesMem[i * nopen + j];
    }
  }
  delete [] singlesMem;
 

  double *rotationMem = new double[ norbs * norbs ];
  for (int i = 0; i < norbs; i++) 
    for (int j = 0; j < norbs; j++)
      rotationMem[i * norbs + j] = 0.;
  dataset_rotation = H5Dopen(file, "/rotation", H5P_DEFAULT);
  status = H5Dread(dataset_rotation, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rotationMem);
  for (int i = 0; i < norbs; i++) {
    for (int j = 0; j < norbs; j++) {
      basisRotation(i, j) = rotationMem[i * norbs + j];
    }
  }
  delete [] rotationMem;
  
  
  double *doublesMem = new double[ nexc * nexc ];
  for (int i = 0; i < nexc; i++) 
    for (int j = 0; j < nexc; j++)
      doublesMem[i * nexc + j] = 0.;
  dataset_doubles = H5Dopen(file, "/doubles", H5P_DEFAULT);
  status = H5Dread(dataset_doubles, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, doublesMem);
  for (int i = 0; i < nexc; i++) {
    for (int j = 0; j < nexc; j++) {
      doubles(i, j) = doublesMem[i * nexc + j];
    }
  }
  delete [] doublesMem;
}


// reads uccsd amplitudes
void readUCCSD(std::array<Eigen::MatrixXd, 2>& singles, std::array<Eigen::MatrixXd, 3>& doubles, Eigen::MatrixXd& basisRotation, std::string fname)
{
  int nocc0 = singles[0].rows(), nopen0 = singles[0].cols();
  int nocc1 = singles[1].rows(), nopen1 = singles[1].cols();
  int norbs = nocc0 + nopen0;
  int nexc0 = nocc0 * nopen0, nexc1 = nocc1 * nopen1;

  hid_t file = (-1), dataset_singles0, dataset_singles1, dataset_doubles0, dataset_doubles1, dataset_doubles2, dataset_rotation;  /* identifiers */
  herr_t status;

  H5E_BEGIN_TRY {
    file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  } H5E_END_TRY
 
  if (file < 0) {
    if (commrank == 0) cout << "Amplitudes not found!" << endl;
    exit(0);
  }
  
  {
    double *singlesMem = new double[ nocc0 * nopen0 ];
    for (int i = 0; i < nocc0; i++) 
      for (int j = 0; j < nopen0; j++)
        singlesMem[i * nopen0 + j] = 0.;
    dataset_singles0 = H5Dopen(file, "/singles0", H5P_DEFAULT);
    status = H5Dread(dataset_singles0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, singlesMem);
    for (int i = 0; i < nocc0; i++) {
      for (int j = 0; j < nopen0; j++) {
        singles[0](i, j) = singlesMem[i * nopen0 + j];
      }
    }
    delete [] singlesMem;
  }
  
  {
    double *singlesMem = new double[ nocc1 * nopen1 ];
    for (int i = 0; i < nocc1; i++) 
      for (int j = 0; j < nopen1; j++)
        singlesMem[i * nopen1 + j] = 0.;
    dataset_singles1 = H5Dopen(file, "/singles1", H5P_DEFAULT);
    status = H5Dread(dataset_singles1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, singlesMem);
    for (int i = 0; i < nocc1; i++) {
      for (int j = 0; j < nopen1; j++) {
        singles[1](i, j) = singlesMem[i * nopen1 + j];
      }
    }
    delete [] singlesMem;
  }

  {
    double *rotationMem = new double[ norbs * norbs ];
    for (int i = 0; i < norbs; i++) 
      for (int j = 0; j < norbs; j++)
        rotationMem[i * norbs + j] = 0.;
    dataset_rotation = H5Dopen(file, "/rotation", H5P_DEFAULT);
    status = H5Dread(dataset_rotation, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rotationMem);
    for (int i = 0; i < norbs; i++) {
      for (int j = 0; j < norbs; j++) {
        basisRotation(i, j) = rotationMem[i * norbs + j];
      }
    }
    delete [] rotationMem;
  }
 
  {
    double *doublesMem = new double[ nexc0 * nexc0 ];
    for (int i = 0; i < nexc0; i++) 
      for (int j = 0; j < nexc0; j++)
        doublesMem[i * nexc0 + j] = 0.;
    dataset_doubles0 = H5Dopen(file, "/doubles0", H5P_DEFAULT);
    status = H5Dread(dataset_doubles0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, doublesMem);
    for (int i = 0; i < nexc0; i++) {
      for (int j = 0; j < nexc0; j++) {
        doubles[0](i, j) = doublesMem[i * nexc0 + j];
      }
    }
    delete [] doublesMem;
  }
  
  {
    double *doublesMem = new double[ nexc1 * nexc1 ];
    for (int i = 0; i < nexc1; i++) 
      for (int j = 0; j < nexc1; j++)
        doublesMem[i * nexc1 + j] = 0.;
    dataset_doubles1 = H5Dopen(file, "/doubles1", H5P_DEFAULT);
    status = H5Dread(dataset_doubles1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, doublesMem);
    for (int i = 0; i < nexc1; i++) {
      for (int j = 0; j < nexc1; j++) {
        doubles[1](i, j) = doublesMem[i * nexc1 + j];
      }
    }
    delete [] doublesMem;
  }
  
  {
    double *doublesMem = new double[ nexc0 * nexc1 ];
    for (int i = 0; i < nexc0; i++) 
      for (int j = 0; j < nexc1; j++)
        doublesMem[i * nexc1 + j] = 0.;
    dataset_doubles2 = H5Dopen(file, "/doubles2", H5P_DEFAULT);
    status = H5Dread(dataset_doubles2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, doublesMem);
    for (int i = 0; i < nexc0; i++) {
      for (int j = 0; j < nexc1; j++) {
        doubles[2](i, j) = doublesMem[i * nexc1 + j];
      }
    }
    delete [] doublesMem;
  }
}
