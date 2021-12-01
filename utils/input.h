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
      & fullRestart
      & wavefunctionType
      & integralsFile
      & ghfDets
      & numResonants
      & singleJastrow
      & readTransOrbs
      & numPermutations
      & maxIter
      & avgIter
      & printLevel
      & debug
      & decay1
      & decay2
      & alpha
      & beta
      & method
      & stochasticIter
      & burnIter
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
      & optimizeCiCoeffs
      & optimizeCps
      & optimizeJastrow
      & optimizeRBM
      & printVars
      & printGrad
      & printJastrow
      & Hamiltonian
      & useLastDet
      & useLogTime
      & ctmc
      & nwalk
      & tau
      & fn_factor
      & nGeneration
      & excitationLevel
      & normSampleThreshold
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
      & actWidth
      & lanczosEpsilon
      & overlapCutoff
      & diagMethod
      & powerShift
      & expCorrelator
      & numHidden
      & maxIterFCIQMC
      & nreplicas
      & nAttemptsEach
      & mainMemoryFac
      & spawnMemoryFac
      & shiftDamping
      & initialShift
      & trialInitFCIQMC
      & minSpawn
      & minPop
      & initialPop
      & initialNDets
      & targetPop
      & initiator
      & initiatorThresh
      & semiStoch
      & semiStochInit
      & semiStochFile
      & uniformExGen
      & heatBathExGen
      & heatBathUniformSingExGen
      & calcEN2
      & useTrialFCIQMC
      & trialWFEstimator
      & importanceSampling
      & applyNodeFCIQMC
      & releaseNodeFCIQMC
      & releaseNodeIter
      & diagonalDumping
      & partialNodeFactor
      & expApprox
      & printAnnihilStats

      // dqmc
      & dt
      & nsteps
      & eneSteps
      & errorTargets
      & fieldStepsize
      & measureFreq
      & orthoSteps
      & ene0Guess
      & numJastrowSamples
      & choleskyThreshold
      & ciThreshold
      & leftWave
      & rightWave
      & ndets
      & phaseless
      // Options related to SC-NEVPT(s):
      & numSCSamples
      & printSCNorms
      & printSCNormFreq
      & readSCNorms
      & continueSCNorms
      & sampleNEVPT2Energy
      & determCCVV
      & efficientNEVPT
      & efficientNEVPT_2
      & exactE_NEVPT
      & NEVPT_writeE
      & NEVPT_readE
      & continueMarkovSCPT
      & stochasticIterNorms
      & stochasticIterEachSC
      & nIterFindInitDets
      & printSCEnergies
      & nWalkSCEnergies
      & SCEnergiesBurnIn
      & SCNormsBurnIn
      & exactPerturber
      & CASEnergy
      & perturberOrb1
      & perturberOrb2
      & fixedResTimeNEVPT_Ene
      & fixedResTimeNEVPT_Norm
      & resTimeNEVPT_Ene
      & resTimeNEVPT_Norm
      & ngrid
      & printFrequency
      & sampleDeterminants;
  }
public:
//General options
  bool restart;                          //option to restart calculation
  bool fullRestart;                          //option to restart calculation
  bool deterministic;                    //Performs a deterministic calculation   
  int printLevel;                        // How much stuff to print
  bool expCorrelator;                    // exponential correlator parameters, to enforce positivity
  bool debug;
  bool ifComplex;                        // breaks and restores complex conjugation symmetry
  bool uagp;                             // brakes S^2 symmetry in uagp
  bool ciCeption;                        // true, when using ci on top of selectedCI
  
  // system options
  std::string integralsFile;            // file containing intergrals, could be text or hdf5 
  int nciCore;                          // number of core spatial orbitals
  int nciAct;                           // number of active spatial orbitals, assumed to be the first in the basis

//input file to define the correlator parts of the wavefunction
  std::string wavefunctionType;
  std::map<int, std::string> correlatorFiles;
  std::string determinantFile;
  int numResonants;
  bool ghfDets;
  double normSampleThreshold;
  bool singleJastrow;
  bool readTransOrbs;
  int numPermutations;

//Used in the stochastic calculation of E and PT evaluation
  int stochasticIter;                    //Number of stochastic steps
  int burnIter;                          //Number of burn in steps
  int integralSampleSize;                //This specifies the number of determinants to sample out of the o^2v^2 possible determinants after the action of V
  size_t seed;                              // seed for the random number generator
  bool detsInCAS;
  double PTlambda;                       // In PT we have to apply H0- E0, here E0 = lambda x <psi0|H0|psi0> + (1 - lambda) x <psi0|H|psi0>
  double epsilon;                        // This is the usual epsilon for the heat bath truncation of integrals
  double screen;                         //This is the screening parameter, any integral below this is ignored
  bool doHessian;                        //This calcules the Hessian and overlap for the linear method
  std::string hf;
  bool optimizeOrbs;
  bool optimizeCiCoeffs;
  bool optimizeCps;
  bool optimizeJastrow;                  //used in jrbm
  bool optimizeRBM;                      //used in jrbm
  bool printVars;
  bool printGrad;
  bool printJastrow;
  HAM Hamiltonian;
  bool useLastDet;                       //stores last det instead of bestdet
  bool useLogTime;                       //uses log sampled time in CTMC

// SC-NEVPT2(s) options:
  bool determCCVV;                       // In NEVPT2 calculations, calculate the CCVV energy by the exact formula
  bool efficientNEVPT;                   // More efficient sampling in the SC-NEVPT2(s) method
  bool efficientNEVPT_2;                 // More efficient sampling in the SC-NEVPT2(s) method -
                                         // a second approach to this sampling
  bool exactE_NEVPT;                     // Follows the efficient approach to SC-NEVPT2(s), but the energies
                                         // E_l^k are all calculated exactly, without statistical error
  bool NEVPT_writeE;                     // These options are used to exactly calculate the energies of all NEVPT2
  bool NEVPT_readE;                      //   perturbers, and print them. The second option can then be
                                         //   used to read them back in again (for example if using different
                                         //   norms with a different seed) without recalculating them
  bool exactPerturber;                   // Exactly calcualte the energy of a perturber in SC-NEVPT2
  double CASEnergy;                      // User can input a CAS energy, for use in the exactPerturber option
  int perturberOrb1;                     // The excited core and virtual (spin) orbitals which define the perturber,
  int perturberOrb2;                     // in an 'exactPerturber' NEVPT2 calculation
  int numSCSamples;                      // When performing SC-NEVPT2 with the efficientNEVPT_2 algorithm, how
                                         // many samples of 1/(E_0-E_l^k) to take?
  bool printSCNorms;                     // Should we print out the norms of strongly contracted states (in SC-NEVPT2)
  int printSCNormFreq;                   // How often should we print out norms of strongly contracted states (for
                                         // printSCNorms option)
  bool readSCNorms;                      // Do not sample SC norms, but instead read them from previously-printed file
  bool continueSCNorms;                  // Read SC norms from files, and then continue sampling them
  bool sampleNEVPT2Energy;               // If true, then perform sampling of the NEVPT2 energy
  bool continueMarkovSCPT;               // In SC-NEVPT2(s), option to store the final det in each sampling of a SC space
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
  int SCNormsBurnIn;                     // For SC-NEVPT2(s), this is the number of iterations used for burn in
                                         //(thrown away), when sampling N_l^k
  bool fixedResTimeNEVPT_Ene;            // If true, estimate E_l^k in SC-NEVPT2 with a fixed residence time.
                                         // Otherwise, use a fixed iteration count
  bool fixedResTimeNEVPT_Norm;           // If true, estimate E_l^k in SC-NEVPT2 with a fixed residence time.
                                         // Otherwise, use a fixed iteration count
  double resTimeNEVPT_Ene;               // For NEVPT2, this is the total residence time for each E_l^k sampling
  double resTimeNEVPT_Norm;              // For NEVPT2, this is the total residence time for each N_l^k sampling

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
  bool usingFOIS; // Is this is a MRCI/MRPT calculation, sampling the FOIS only
  double actWidth; //used in lanczos
  double lanczosEpsilon; //used in lanczos
  double overlapCutoff; //used in SCCI
  std::string diagMethod;
  double powerShift;
  double ciThreshold;

  // Options for FCIQMC
  int maxIterFCIQMC;
  int nreplicas;
  int nAttemptsEach;
  double shiftDamping;
  double mainMemoryFac;
  double spawnMemoryFac;
  double initialShift;
  double minSpawn;
  double minPop;
  double initialPop;
  int initialNDets;
  bool trialInitFCIQMC;
  double targetPop;
  bool initiator;
  double initiatorThresh;
  bool semiStoch;
  bool semiStochInit;
  std::string semiStochFile;
  bool uniformExGen;
  bool heatBathExGen;
  bool heatBathUniformSingExGen;
  bool calcEN2;
  bool useTrialFCIQMC;
  bool trialWFEstimator;
  bool importanceSampling;
  bool applyNodeFCIQMC;
  bool releaseNodeFCIQMC;
  int releaseNodeIter;
  bool diagonalDumping;
  double partialNodeFactor;
  bool expApprox;
  bool printAnnihilStats;

  //options for rbm
  int numHidden;

  // options for dqmc
  size_t nsteps;
  std::vector<int> eneSteps;
  std::vector<double> errorTargets;
  double dt;
  double fieldStepsize;
  size_t measureFreq;
  size_t orthoSteps;
  double ene0Guess;
  size_t numJastrowSamples;
  int ngrid;
  size_t printFrequency;
  int sampleDeterminants;
  double choleskyThreshold;
  std::string leftWave;
  std::string rightWave;
  size_t ndets;
  bool phaseless;
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

void writeMat(Eigen::MatrixXcd& mat, std::string fileName);

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

// for vmc
// reads determinants from Dice, for now assumes rhf dets and converts them into ghf = block_diag(rhf, rhf) 
// the reference determinant, assumed to be the first in the file, is read in as a list of integers
// the rest are stored as excitations from ref
// assumes Dice parity included in ci coeffs
// the parity vector in the function arguments refers to parity of excitations required when using matrix det lemma
void readDeterminants(std::string input, std::vector<int>& ref, std::vector<int>& open, std::vector<std::array<Eigen::VectorXi, 2>>& ciExcitations,
        std::vector<int>& ciParity, std::vector<double>& ciCoeffs);


void readDeterminantsGHF(std::string input, std::vector<int>& ref, std::vector<int>& open, std::vector<std::array<Eigen::VectorXi, 2>>& ciExcitations,
        std::vector<int>& ciParity, std::vector<double>& ciCoeffs);


// for dqmc
// reads determinants from Dice, uses uhf dets
// the reference determinant, assumed to be the first in the file, is read in as a list of integers
// the rest are stored as excitations from ref
// assumes Dice parity included in ci coeffs
// the parity vector in the function arguments refers to parity of excitations required when using matrix det lemma
void readDeterminants(std::string input, std::array<std::vector<int>, 2>& ref, std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2>& ciExcitations,
        std::vector<double>& ciParity, std::vector<double>& ciCoeffs);

// same as above but for binary files
void readDeterminantsBinary(std::string input, std::array<std::vector<int>, 2>& ref, std::array<std::vector<std::array<Eigen::VectorXi, 2>>, 2>& ciExcitations,
        std::vector<double>& ciParity, std::vector<double>& ciCoeffs);


void readSpinRDM(std::string fname, Eigen::MatrixXd& oneRDM, Eigen::MatrixXd& twoRDM);


// reads ccsd amplitudes
void readCCSD(Eigen::MatrixXd& singles, Eigen::MatrixXd& doubles, Eigen::MatrixXd& basisRotation, std::string fname = "ccsd.h5");


// reads uccsd amplitudes
void readUCCSD(std::array<Eigen::MatrixXd, 2>& singles, std::array<Eigen::MatrixXd, 3>& doubles, Eigen::MatrixXd& basisRotation, std::string fname = "uccsd.h5");
#endif
