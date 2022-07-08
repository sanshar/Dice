#ifndef fnn_HEADER_H
#define fnn_HEADER_H
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <algorithm>
//#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <boost/serialization/serialization.hpp>

using namespace std;
using namespace Eigen;

class fnn
{ // feed forward neural network
  
  private:
    
    friend class boost::serialization::access;
    template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        ar & numLayers
          & sizes
          & weights
          & biases;
      }
  
  public:
    
    int numLayers;
    vector<int> sizes;             // number of neurons in each layer
    vector<MatrixXd> weights;
    vector<VectorXd> biases;
    
    // constructor
    fnn() {};
    
    fnn(vector<int> pSizes) {
      numLayers = pSizes.size();
      sizes = pSizes;
      for (int i = 1; i < numLayers; i++) {
        weights.push_back(MatrixXd::Random(sizes[i], sizes[i - 1])/pow(sizes[i - 1]*sizes[i], 1));
        biases.push_back(VectorXd::Random(sizes[i])/pow(sizes[i], 1));
      }
    }

    // constructor
    fnn(vector<int> pSizes, vector<MatrixXd>& pWeights, vector<VectorXd>& pBiases) {
      numLayers = pSizes.size();
      sizes = pSizes;
      weights = pWeights;
      biases = pBiases;
    }
    
    // number of variables : weights and biases
    size_t getNumVariables() const {
      size_t num = 0;
      for (int i = 1; i < numLayers; i++) num += sizes[i - 1] * sizes[i] + sizes[i];
      return num;
    }

    // all weights followed by all biases
    void getVariables(VectorXd& vars) const {
      vars = VectorXd::Zero(getNumVariables());
      size_t counter  = 0;
      
      // weights
      for (int n = 0; n < numLayers - 1; n++) {
        for (int j = 0; j < sizes[n]; j++) {
          for (int i = 0; i < sizes[n + 1]; i++) {
            vars[counter] = weights[n](i, j);
            counter++;
          }
        }
      }
      
      // biases
      for (int n = 0; n < numLayers - 1; n++) {
        for (int i = 0; i < sizes[n + 1]; i++) {
          vars[counter] = biases[n](i);
          counter++;
        }
      }
    }
    
    // all weights followed by all biases
    void updateVariables(const VectorXd& vars) {
      size_t counter  = 0;
      // weights
      for (int n = 0; n < numLayers - 1; n++) {
        for (int j = 0; j < sizes[n]; j++) {
          for (int i = 0; i < sizes[n + 1]; i++) {
            weights[n](i, j) = vars[counter];
            counter++;
          }
        }
      }
      
      // biases
      for (int n = 0; n < numLayers - 1; n++) {
        for (int i = 0; i < sizes[n + 1]; i++) {
          biases[n](i) = vars[counter];
          counter++;
        }
      }
    }

    // activation function
    static double activation(double input) {
      // ramp
      return (input > 0) ? input : 0.;
    }
    
    // derivative of activation function
    static double dactivation(double input) {
      // ramp
      return (input > 0) ? 1. : 0.;
    }

    // forward propagation
    void feedForward(const VectorXd& input, VectorXd& output) const {
      output = input;
      for (int i = 0; i < numLayers - 1; i++) 
        output = (weights[i] * output + biases[i]).unaryExpr(std::ref(activation));
    }
    
    // evaluate the nn for given input and cost function
    double evaluate(const VectorXd& input, std::function<double(VectorXd&)> cost) const {
      VectorXd output;
      feedForward(input, output);
      return cost(output);
    }
   
    // gradient evaluation
    void backPropagate(const VectorXd& input, std::function<void(VectorXd&, VectorXd&)> costGradient, VectorXd& grad) const {
      // forward
      vector<VectorXd> zVecs, aVecs; // intermediates created during feedforward
      zVecs.push_back(input);
      aVecs.push_back(input);
      for (int i = 0; i < numLayers - 1; i++) {
        VectorXd zVec = weights[i] * aVecs[i] + biases[i];
        zVecs.push_back(zVec);
        VectorXd aVec = (zVec).unaryExpr(std::ref(activation));
        aVecs.push_back(aVec);
      }

      // backward 
      vector<MatrixXd> weightDerivatives;
      vector<VectorXd> biasDerivatives;
      VectorXd costGradVec;
      costGradient(aVecs[numLayers - 1], costGradVec);
      VectorXd delta = costGradVec.cwiseProduct((zVecs[numLayers - 1]).unaryExpr(std::ref(dactivation)));
      // these are built in reverse order
      biasDerivatives.push_back(delta);
      weightDerivatives.push_back(delta * aVecs[numLayers - 2].transpose());
      for (int i = 2; i < numLayers; i++) { // l = numLayers - i - 1
        delta = (weights[numLayers - i].transpose() * delta).cwiseProduct((zVecs[numLayers - i]).unaryExpr(std::ref(dactivation)));
        biasDerivatives.push_back(delta);
        weightDerivatives.push_back(delta * aVecs[numLayers - i - 1].transpose());
      }
      // reversing to correct order
      reverse(biasDerivatives.begin(), biasDerivatives.end());
      reverse(weightDerivatives.begin(), weightDerivatives.end());
   
      // flattening into grad vector
      grad = VectorXd::Zero(getNumVariables());
      size_t counter  = 0;
      
      // weights
      for (int n = 0; n < numLayers - 1; n++) {
        for (int j = 0; j < sizes[n]; j++) {
          for (int i = 0; i < sizes[n + 1]; i++) {
            grad[counter] = weightDerivatives[n](i, j);
            counter++;
          }
        }
      }


      // biases
      for (int n = 0; n < numLayers - 1; n++) {
        for (int i = 0; i < sizes[n + 1]; i++) {
          grad[counter] = biasDerivatives[n](i);
          counter++;
        }
      }
      
    }

    // print for debugging
    friend ostream& operator<<(ostream& os, const fnn& nn) {
      os << "numLayers  " << nn.numLayers << endl;
      os << "sizes  "; 
      for (int i : nn.sizes) os << i << "  "; 
      os << endl << endl;
      os << "weights\n\n"; 
      for (MatrixXd i : nn.weights) os << i << endl << endl;
      os << "biases\n\n"; 
      for (MatrixXd i : nn.biases) os << i << endl << endl;
      return os;
    }
};

#endif
