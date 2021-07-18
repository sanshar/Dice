#ifndef CDFCI_H
#define CDFCI_H
#include <vector>
#include <unordered_map>
#include <array>
#include <set>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "global.h"
#include "Determinants.h"
#include <iomanip>
#include "robin_hood.h"
#include "input.h"

#define float128 __float128

using namespace std;
using namespace Eigen;

class Determinant;
class oneInt;
class twoInt;
class twoIntHeatBath;
class twoIntHeatBathSHM;
class schedule;

template<>
struct std::hash<Determinant> {
  const std::size_t operator()(const Determinant& det) const {
    //Determinant this_det = det;
    //return det.repr[0];
    //return det.repr[0] + det.repr[1]*179426549;
    //return det.repr[0] * 2038076783 + det.repr[1] * 179426549 + det.repr[2] * 500002577;
    return det.repr[0] * 2038076783 + det.repr[1] * 179426549 + det.repr[2] * 500002577 + det.repr[3] * 255477023;
  }
};

namespace cdfci {
  // currently assumes real ci vector
  using value_type = std::pair<Determinant, array<double, 2>>;
  using hash_det = robin_hood::unordered_flat_map<Determinant, array<double, 2>, std::hash<Determinant>, std::equal_to<Determinant>>;
  using DetToIndex = robin_hood::unordered_node_map<Determinant, int, std::hash<Determinant>, std::equal_to<Determinant>>;
  // retain the same function call as DoVariational in SHCIbasics.
  // use hash map for determinants internally
  vector<double> DoVariational(vector<MatrixXx>& ci, vector<Determinant> & Dets, schedule& schd, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, oneInt& I1, double& coreE, int nelec, bool DoRDM=false);
  
  void getDeterminantsVariational( Determinant& d, double epsilon, CItype ci1, CItype ci2,
        oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb,
        vector<int>& irreps, double coreE, double E0,
        DetToIndex& det_to_index,
        schedule& schd, int Nmc, int nelec);
  
  set<Determinant> sampleExtraEntry(hash_det& wfn, int nelec);
  
  void cdfciSolver(hash_det& wfn,  Determinant& hf, schedule& schd, pair<double, double>& ene, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, double& coreE, vector<double> & E0, int nelec, double thresh=1e10, bool sample=false);
  
  value_type CoordinatePickGcdGrad(vector<value_type> sub_dets, double norm);
  
  double CoordinateUpdate(value_type& det_picked, hash_det & wfn, pair<double, double> & ene, vector<double> E0, oneInt& I1, twoInt& I2, double& coreE);
  
  vector<value_type> getSubDets(value_type& det, hash_det& wfn, int nelec, bool sample=false);

  void civectorUpdate(vector<value_type> &column, hash_det& wfn, double dx, pair<double, double>& ene, oneInt& I1, twoInt& I2, double& coreE, double thresh=1e10, bool sample=false);
  
  hash_det precondition(pair<double, double>& ene, vector<Determinant>& dets, vector<MatrixXx> &ci, vector<double> energy, oneInt& I1, twoInt& I2, double& coreE, double thresh=1e10, bool sample=false);

  vector<double> compute_residual(vector<double>& x, vector<double>& z, vector<pair<float128, float128>>& ene);
  double compute_residual(vector<double>& x, vector<double>& z, vector<pair<float128, float128>>& ene, int& iroot);
  void solve(schedule& schd, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, double& coreE, vector<double>& E0, vector<MatrixXx>& ci, vector<Determinant>& dets);
  void sequential_solve(schedule& schd, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, double& coreE, vector<double>& E0, vector<MatrixXx>& ci, vector<Determinant>& dets);
}
#endif
