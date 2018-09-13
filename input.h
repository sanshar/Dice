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
enum Method { sgd, nestorov, rmsprop, adam, amsgrad };
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
      & davidsonPrecondition
      & diisSize
      & maxIter
      & printLevel
      & gradientFactor
      & mingradientFactor
      & method
      & stochasticIter
      & integralSampleSize
      & momentum
      & momentumDecay
      & decay
      & learningEpoch
      & seed
      & PTlambda
      & epsilon
      & screen
      & determinantFile
      & doHessian
      & uhf
      & optimizeOrbs
      & Hamiltonian
      & nwalk
      & tau
      & fn_factor
      & nGeneration
      & excitationLevel
      & optvar
      & gd
      & sr;
  }
public:
//General options
  bool restart;                          //option to restart calculation
  bool deterministic;                    //Performs a deterministic calculation   
  int printLevel;                        // How much stuff to print


//input file to define the correlator parts of the wavefunction
  std::map<int, std::string> correlatorFiles;

  std::string determinantFile;

//Used in the stochastic calculation of E and PT evaluation
  int stochasticIter;                    //Number of stochastic steps
  int integralSampleSize;                //This specifies the number of determinants to sample out of the o^2v^2 possible determinants after the action of V
  int seed;                              // seed for the random number generator
  double PTlambda;                       // In PT we have to apply H0- E0, here E0 = lambda x <psi0|H0|psi0> + (1 - lambda) x <psi0|H|psi0>
  double epsilon;                        // This is the usual epsilon for the heat bath truncation of integrals
  double screen;                         //This is the screening parameter, any integral below this is ignored
  bool doHessian;                        //This calcules the Hessian and overlap for the linear method
  bool uhf;
  bool optimizeOrbs;
  HAM Hamiltonian;
  bool optvar;
  bool sr;
  bool gd;

//Deprecated options for optimizers
//because now we just use the python implementation
  double tol;                          
  bool davidsonPrecondition;
  int diisSize;
  int maxIter;
  double gradientFactor;
  double mingradientFactor;
  Method method;
  double momentum;
  double momentumDecay;
  double decay;
  int learningEpoch;


  //options for gfmc
  int nwalk;
  double tau;
  double fn_factor;
  int nGeneration;

  //option for configuration interaction
  int excitationLevel;
};

/**
 * This just gives the matrix of MO coefficients, right now
 * we just assume RHF determinant, but later this can be generalized to 
 * an alpha and a beta matrix
 * params:
 *   Matrix:  this is the matrix of the mo coefficients
 */
void readHF(Eigen::MatrixXd& hforbsA, Eigen::MatrixXd& hforbsB, bool uhf);

/**
 * Reads the input file which by default is input.dat, but can be anything
 * else that is specified on the command line
 * 
 * params:  
 *    input:    the input file (unchanged)
 *    schd :    this is the object of class schedule that is populated by the options
 *    print:    How much to print
 */
void readInput(const std::string input, schedule& schd, bool print=true);

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


/**
 * We are just reading the set of determinants and their ci coefficients
 * for the multi-slater part of the multi-slater Jastrow wavefunction
 */
void readDeterminants(std::string input, std::vector<Determinant>& determinants,
        std::vector<double>& ciExpansion);
#endif
