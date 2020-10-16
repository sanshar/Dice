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

using namespace std;
using namespace Eigen;
//using namespace Eigen;

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
  using dcomplex = std::complex<double>;
  using value_type = std::pair<Determinant, array<complex<double>, 2>>;
  using hash_det = robin_hood::unordered_flat_map<Determinant, array<complex<double>, 2>, std::hash<Determinant>, std::equal_to<Determinant>>;
  // retain the same function call as DoVariational in SHCIbasics.
  // use hash map for determinants internally
  set<Determinant> sampleExtraEntry(hash_det& wfn, int nelec);
  void cdfciSolver(hash_det& wfn,  Determinant& hf, schedule& schd, pair<dcomplex, double>& ene, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, double& coreE, vector<double> & E0, int nelec, double thresh=1e10, bool sample=false);
  void cyclicSolver(hash_det& wfn,  vector<Determinant>& Dets, schedule& schd, pair<dcomplex, double>& ene, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, double& coreE, vector<double> & E0, int nelec, double thresh=1e10, bool sample=false);
  value_type CoordinatePickGcdGrad(vector<value_type> sub_dets, double norm, bool real_part);
  dcomplex CoordinateUpdate(value_type& det_picked, hash_det & wfn, pair<dcomplex, double> & ene, vector<double> E0, oneInt& I1, twoInt& I2, double& coreE, bool real_part);
  vector<value_type> getSubDets(value_type& det, hash_det& wfn, int nelec, bool sample=false);
  void civectorUpdate(vector<value_type> &column, hash_det& wfn, dcomplex dx, pair<dcomplex, double>& ene, oneInt& I1, twoInt& I2, double& coreE, double thresh=1e10, bool sample=false);
  hash_det precondition(pair<dcomplex, double>& ene, vector<Determinant>& dets, vector<MatrixXx> &ci, vector<double> energy, oneInt& I1, twoInt& I2, double& coreE, double thresh=1e10, bool sample=false);
}
#endif
