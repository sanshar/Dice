#pragma once
#include <vector>
#include <iostream>

using namespace std;

struct tensor {

  vector<int> dimensions;
  double* vals;
  vector<double> values;
  vector<int> strides;
  
  tensor(vector<int> pdimensions)
  {
    int n = pdimensions.size();
    dimensions.resize(n);
    strides.resize(n);
    int size = 1.;
    for (int i=0; i<n; i++) {
      dimensions[i] = pdimensions[i];
      strides[n-1-i] = i == 0 ? 1 : strides[n-i] * pdimensions[i-1];
      size *= dimensions[i];
    }
    values.resize(size);
    vals = &values[0];

  }
  tensor(vector<int> pdimensions, double* pvals)
  {
    int n = pdimensions.size();
    dimensions.resize(n);
    vals = pvals;
  }

  double& operator()(int i) {
    return vals[i];
  }
  double& operator()(int i, int j) {
    return vals[i*dimensions[1]+j];
  }
  double& operator()(int i, int j, int k) {
    return vals[ (i*dimensions[1]+j)*dimensions[2]+k];
  }
  double& operator()(int i, int j, int k, int l) {
    return vals[ ((i*dimensions[1]+j)*dimensions[2]+k)*dimensions[3]+l];
  }

  const double& operator()(int i) const {
    return vals[i];
  }
  const double& operator()(int i, int j) const {
    return vals[i*dimensions[1]+j];
  }
  const double& operator()(int i, int j, int k) const {
    return vals[ (i*dimensions[1]+j)*dimensions[2]+k];
  }
  const double& operator()(int i, int j, int k, int l) const {
    return vals[ ((i*dimensions[1]+j)*dimensions[2]+k)*dimensions[3]+l];
  }

  void setZero() {
    int size = 1;
    for (int i=0; i<dimensions.size(); i++)
      size *= dimensions[i];

    for (int i=0; i<size; i++)
      vals[i] = 0.0;
  }

};


int contract_IJK_IJL_LK(tensor *O, tensor *C, tensor *S, double scale=1.0, double beta=0.);
