#include "Davidson.h"
#include "Hmult.h"
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

void precondition(MatrixXd& r, MatrixXd& diag, double& e) {
  for (int i=0; i<r.rows(); i++) {
    if (abs(e-diag(i,0)) > 1e-5)
      r(i,0) = r(i,0)/(e-diag(i,0));
    else
      r(i,0) = r(i,0)/(e-diag(i,0)-0.001);
  }
}

