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
#ifndef INPUT_HEADER_H
#define INPUT_HEADER_H
#include <Eigen/Dense>
#include <string>
#include <map>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>

class Correlator;
class Determinant;
enum Method { sgd, amsgrad, ftrl, amsgrad_sgd, sr, linearmethod };
enum HAM {HUBBARD, ABINITIO};


/**
 * This stores all the input options
 * */

struct schedule {
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & restart & deterministic
      & tol & correlatorFiles
      & wavefunctionType
      & maxIter
      & avgIter
      & printLevel
      & printSCNorms
      & printSCNormFreq
      & readSCNorms
      & sampleNEVPT2Energy
      & debug
      & decay1
      & decay2
      & alpha
      & beta
      & method
      & stochasticIter
      & numSCSamples
      & _sgdIter
      & momentum
      & integralSampleSize
      & seed
      & PTlambda
      & epsilon
      & screen
      & determinantFile
      & detsInCAS
      & doHessian
      & hf
      & optimizeOrbs
      & optimizeCps
      & optimizeJastrow
      & optimizeRBM
      & printVars
      & printGrad
      & Hamiltonian
      & ctmc
      & nwalk
      & tau
      & fn_factor
      & nGeneration
      & excitationLevel
      & numActive
      & nciCore
      & nciAct
      & usingFOIS
      & sDiagShift
      & cgIter
      & stepsize
      & ifComplex
      & uagp
      & ciCeption
      & determCCVV
      & efficientNEVPT
      & efficientNEVPT_2
      & actWidth
      & overlapCutoff
      & diagMethod
      & powerShift
      & expCorrelator
      & nAttemptsEach
      & mainMemoryFac
      & spawnMemoryFac
      & shiftDamping
      & initialShift
      & minSpawn
      & minPop
      & initialPop
      & targetPop
      & numHidden
      & continueMarkovSCPT
      & stochasticIterNorms
      & stochasticIterEachSC
      & nIterFindInitDets
      & printSCEnergies
      & nWalkSCEnergies
      & SCEnergiesBurnIn
      & NEVPTBiasCorrection;
  }
public:
//General options
  bool restart;                          //option to restart calculation
  bool deterministic;                    //Performs a deterministic calculation   
  int printLevel;                        // How much stuff to print
  bool expCorrelator;                    //exponential correlator parameters, to enforce positivity
  bool debug;
  bool ifComplex;                        //breaks and restores complex conjugation symmetry 
  bool uagp;                             //brakes S^2 symmetry in uagp
  bool ciCeption;                        //true, when using ci on top of selectedCI
  bool determCCVV;                       //In NEVPT2 calculations, calculate the CCVV energy by the exact formula
  bool efficientNEVPT;                   //More efficient sampling in the SC-NEVPT2(s) method
  bool efficientNEVPT_2;                 //More efficient sampling in the SC-NEVPT2(s) method - a second approach to this sampling

//input file to define the correlator parts of the wavefunction
  std::string wavefunctionType;
  std::map<int, std::string> correlatorFiles;
  std::string determinantFile;
  bool detsInCAS;

//Used in the stochastic calculation of E and PT evaluation
  int stochasticIter;                    // Number of stochastic steps
  int numSCSamples;                      // When performing SC-NEVPT2 with the efficientNEVPT_2 algorithm, how many samples of 1/(E_0-E_l^k) to take?
  int integralSampleSize;                // This specifies the number of determinants to sample out of the o^2v^2 possible determinants after the action of V
  int seed;                              // seed for the random number generator
  double PTlambda;                       // In PT we have to apply H0- E0, here E0 = lambda x <psi0|H0|psi0> + (1 - lambda) x <psi0|H|psi0>
  double epsilon;                        // This is the usual epsilon for the heat bath truncation of integrals
  double screen;                         //This is the screening parameter, any integral below this is ignored
  bool doHessian;                        //This calcules the Hessian and overlap for the linear method
  std::string hf;
  bool optimizeOrbs;
  bool optimizeCps;
  bool optimizeJastrow;//used in jrbm
  bool optimizeRBM;//used in jrbm
  bool printVars;
  bool printGrad;
  bool printSCNorms;                     // Should we print out the norms of strongly contracted states (in SC-NEVPT2)
  int printSCNormFreq;                   // How often should we print out norms of strongly contracted states (for printSCNorms option)
  bool readSCNorms;                      // Do not sample SC norms, but instead read them from previously-printed file
  bool sampleNEVPT2Energy;               // If true, then perform sampling of the NEVPT2 energy
  bool continueMarkovSCPT;               // In SC-NEVPT2(s), option to store the final det in each sampling of a SC space
  HAM Hamiltonian;
  int stochasticIterNorms;               // Number of stochastic steps when calculating norms of SC states,
                                         // for the efficientNEVPT option
  int stochasticIterEachSC;              // Number of stochastic steps for each strongly contracted (SC) state,
                                         // for the efficientNEVPT option
  int nIterFindInitDets;                 // The number of iterations used to find initial determinants for
                                         // SC-NEVPT2(s) calculations
  bool printSCEnergies;                  // In SC-NEVPT2(s), print individual samples for the sampling of E_l^k.
  int nWalkSCEnergies;                   // If printSCEnergies = true, then this specifies how many walkers to
                                         // use when sampling E_l^k
  int SCEnergiesBurnIn;                  // For SC-NEVPT2(s), this is the number of iterations used for burn in
                                         //(thrown away), when sampling E_l^k
  bool NEVPTBiasCorrection;              // If true, include correction to account for error in E[1/x] expectation value

//Deprecated options for optimizers
//because now we just use the python implementation
  double tol;  
  double stepsize;
  double decay1;
  double decay2;
  double alpha;
  double beta;
  double momentum;
  int maxIter;                     
  int avgIter;                     
  int _sgdIter;
  Method method;
  double sDiagShift;
  int cgIter;
  bool ctmc;

  /*
  bool davidsonPrecondition;
  int diisSize;
  double gradientFactor;
  double mingradientFactor;
  double momentum;
  double momentumDecay;
  double decay;
  int learningEpoch;
  */

  //options for gfmc
  int nwalk;
  double tau;
  double fn_factor;
  int nGeneration;

  //options for configuration interaction
  int excitationLevel;
  int numActive; //number of active spatial orbitals, assumed to be the first in the basis
  int nciCore; //number of core spatial orbitals
  int nciAct; //number of active spatial orbitals, assumed to be the first in the basis
  bool usingFOIS; // Is this is a MRCI/MRPT calculation, sampling the FOIS only
  double actWidth; //used in lanczos
  double overlapCutoff; //used in SCCI
  std::string diagMethod;
  double powerShift;

  //options for FCIQMC
  int nAttemptsEach;
  double shiftDamping;
  double mainMemoryFac;
  double spawnMemoryFac;
  double initialShift;
  double minSpawn;
  double minPop;
  double initialPop;
  double targetPop;

  //options for rbm
  int numHidden;
};

/**
 * This reads the matrix of MO coefficients from 'hf.txt'
 * an alpha and a beta matrix
 * params:
 *   Matrices: matrices of the mo coefficients (nxn for rhf and uhf, 2nx2n for ghf)
 *   hf string: rhf, uhf or ghf
 */
void readHF(Eigen::MatrixXd& hforbsA, Eigen::MatrixXd& hforbsB, std::string hf);

/**
 * This reads the pairing matrix from 'pairMat.txt'
 * params:
 *   Matrix: matrix to be read into
 */
void readPairMat(Eigen::MatrixXd& pairMat);

/**
 * Reads the input file which by default is input.dat, but can be anything
 * else that is specified on the command line
 * 
 * params:  
 *    input:    the input file (unchanged)
 *    schd :    this is the object of class schedule that is populated by the options
 *    print:    How much to print
 */

void readMat(Eigen::MatrixXd& mat, std::string fileName);

void readMat(Eigen::MatrixXcd& mat, std::string fileName);

void readInput(const std::string inputFile, schedule& schd, bool print=true);

/**
 * We need information about the correlators because the wavefunction is
 * |Psi> = J|D>, where J is the set of jastro factors (correlators)
 * The correlator file just contains the tuple of sites that form the correlators
 * For instance, for two site correlators the file will contain lines just tell you the 
 * orbitals , e.g.
 * 0 1
 * 2 3 ... 
 * 
 * params: 
 *    input:          the input file (unchanged)
 *    correlatorSize: the size of the correlator (unchanged)
 *    correlators   : the vector of correlators,
 *                    its usually empty at input and then is filled with Correlators
 */
void readCorrelator(std::string input, int correlatorSize,
		    std::vector<Correlator>& correlators);

void readCorrelator(const std::pair<int, std::string>& p,
		    std::vector<Correlator>& correlators);


/**
 * We are just reading the set of determinants and their ci coefficients
 * for the multi-slater part of the multi-slater Jastrow wavefunction
 */
void readDeterminants(std::string input, std::vector<Determinant>& determinants,
        std::vector<double>& ciExpansion);
#endif
