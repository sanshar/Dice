#pragma once
#include <boost/math/special_functions/legendre.hpp>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace boost;

//gives (l-m)!/(l+m)!
double factorialRatio(int l, int m);

struct CalculateSphHarmonics {
  int lmax;
  Eigen::VectorXd values;

  CalculateSphHarmonics() : lmax(0) {
    values.resize( (lmax+1)*(lmax+1));
    values.setZero();
  }
  
  CalculateSphHarmonics(int plmax) : lmax(plmax)
  {
    values.resize( (lmax+1)*(lmax+1));
    values.setZero();
  }

  double& getval(int l, int m) {
    size_t index = l*l + (l + m);
    return values[index];
  }

  //calculate all the spherical harmonics at theta phi
  void populate(double theta, double phi);
};


    
