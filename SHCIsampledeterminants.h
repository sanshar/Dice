#ifndef SHCI_SAMPLEDETERMINANTS_H
#define SHCI_SAMPLEDETERMINANTS_H
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <list>
#include <tuple>
#include <map>

using namespace std;
using namespace Eigen;
#include "global.h"

namespace SHCIsampledeterminants {


  int sample_round(MatrixXx& ci, double eps, std::vector<int>& Sample1, std::vector<CItype>& newWts);
  void setUpAliasMethod(MatrixXx& ci, double& cumulative, std::vector<int>& alias, std::vector<double>& prob) ;
  int sample_N2_alias(MatrixXx& ci, double& cumulative, std::vector<int>& Sample1, std::vector<CItype>& newWts, std::vector<int>& alias, std::vector<double>& prob) ;
  int sample_N2_withoutalias(MatrixXx& ci, double& cumulative, std::vector<int>& Sample1, std::vector<CItype>& newWts);
  int sample_N(MatrixXx& ci, double& cumulative, std::vector<int>& Sample1, std::vector<CItype>& newWts);
};

#endif
