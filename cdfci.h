#ifndef CDFCI_H
#define CDFCI_H
#include <vector>
#include <unordered_map>
#include <array>
#include <set>
#include "global.h"
#include "Determinants.h"
#include <iomanip>
#include "robin_hood.h"

using namespace std;
using namespace Eigen;

class Determinant;
class oneInt;
class twoInt;
class twoIntHeatBath;
class twoIntHeatBathSHM;
class schedule;

namespace cdfci {
  // currently assumes real ci vector
  using value_type = std::pair<Determinant, array<double, 2>>;
  using hash_det = robin_hood::unordered_flat_map<Determinant, array<double, 2>, std::hash<Determinant>, std::equal_to<Determinant>>;
  // retain the same function call as DoVariational in SHCIbasics.
  // use hash map for determinants internally
  vector<double> DoVariational(vector<MatrixXx>& ci, vector<Determinant> & Dets, schedule& schd, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, oneInt& I1, double& coreE, int nelec, bool DoRDM=false);
  void getDeterminantsVariational( Determinant& d, double epsilon, CItype ci1, CItype ci2,
        oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb,
        vector<int>& irreps, double coreE, double E0,
        hash_det& wavefunction, set<Determinant>& new_dets,
        schedule& schd, int Nmc, int nelec);
  void cdfciSolver(hash_det& wfn, set<Determinant>& new_dets, Determinant& hf, pair<double, double>& ene, oneInt& I1, twoInt& I2, double& coreE, vector<double> & E0, int nelec);
  value_type CoordinatePickGcdGrad(vector<value_type> sub_dets, double norm);
  double CoordinateUpdate(value_type& det_picked, hash_det & wfn, pair<double, double> & ene, vector<double> E0, oneInt& I1, twoInt& I2, double& coreE);
  vector<value_type> getSubDets(value_type& det, hash_det& wfn, int nelec);
  void civectorUpdate(vector<value_type> &column, hash_det& wfn, double dx, pair<double, double>& ene, oneInt& I1, twoInt& I2, double& coreE);
}
#endif