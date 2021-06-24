#include "cdfci.h"
#include "input.h"
#include "math.h"
#include "SHCIbasics.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <map>
#include <tuple>
#include <vector>
#include "omp.h"
#include "Determinants.h"
#include "SHCIgetdeterminants.h"
#include "SHCISortMpiUtils.h"
#include "SHCItime.h"
#include "input.h"
#include "integral.h"
#include "math.h"
#include "communicate.h"
#include <boost/format.hpp>
#include <complex>

using namespace std;
using namespace Eigen;
using namespace boost;
using namespace cdfci;
using StitchDEH = SHCISortMpiUtils::StitchDEH;
using cdfci::value_type;
typedef unordered_map<Determinant, array<dcomplex, 2>> hash_det;

void cdfci::getDeterminantsVariational(
        Determinant& d, double epsilon, CItype ci1, CItype ci2,
        oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb,
        vector<int>& irreps, double coreE, double E0,
        robin_hood::unordered_set<Determinant>& old_dets,
        robin_hood::unordered_set<Determinant>& new_dets,
        schedule& schd, int Nmc, int nelec) {
//-----------------------------------------------------------------------------
    /*!
    Make the int represenation of open and closed orbitals of determinant
    this helps to speed up the energy calculation

    :Inputs:

        Determinant& d:
            The reference |D_i>
        double epsilon:
            The criterion for chosing new determinants (understood as epsilon/c_i)
        CItype ci1:
            The reference CI coefficient c_i
        CItype ci2:
            The reference CI coefficient c_i
        oneInt& int1:
            One-electron tensor of the Hamiltonian
        twoInt& int2:
            Two-electron tensor of the Hamiltonian
        twoIntHeatBathSHM& I2hb:
            The sorted two-electron integrals to choose the bi-excited determinants
        vector<int>& irreps:
            Irrep of the orbitals
        double coreE:
            The core energy
        double E0:
            The current variational energy
        std::vector<Determinant>& dets:
            The determinants' determinant
        schedule& schd:
            The schedule
        int Nmc:
            BM_description
        int nelec:
            Number of electrons
    */
//-----------------------------------------------------------------------------

  // initialize variables
  int norbs   = d.norbs;
  int nclosed = nelec;
  int nopen   = norbs-nclosed;
  vector<int> closed(nelec,0);
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);

  // mono-excited determinants
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (closed[i] < schd.ncore || open[a] >= schd.ncore+schd.nact) continue;
    CItype integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);


    // generate determinant if integral is above the criterion
    if (std::abs(integral) > epsilon ) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      if(old_dets.find(di) == old_dets.end() && new_dets.find(di) == new_dets.end()) {
        new_dets.emplace(di);
      }
      Determinant detcpy(di);
      detcpy.flipAlphaBeta();
      if(old_dets.find(detcpy) == old_dets.end() && new_dets.find(detcpy) == new_dets.end()) {
        new_dets.emplace(detcpy);
      }
    }
  } // ia

  // bi-excitated determinants
  if (std::abs(int2.maxEntry) < epsilon) return;
  // for all pairs of closed
  for (int ij=0; ij<nclosed*nclosed; ij++) {
    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i], J = closed[j];
    int X = max(I, J), Y = min(I, J);

    if (I < schd.ncore || J < schd.ncore) continue;

    int pairIndex = X*(X+1)/2+Y;
    size_t start = I2hb.startingIndicesIntegrals[pairIndex];
    size_t end   = I2hb.startingIndicesIntegrals[pairIndex+1];
    std::complex<double>* integrals  = I2hb.integrals;
    short* orbIndices = I2hb.pairs;
    // for all HCI integrals
    for (size_t index=start; index<end; index++) {
      // if we are going below the criterion, break
      if (abs(integrals[index]) < epsilon) break;

      // otherwise: generate the determinant corresponding to the current excitation
      int a = orbIndices[2*index], b = orbIndices[2*index+1];
      //if (a/2 >= schd.ncore+schd.nact || b/2 >= schd.ncore+schd.nact) continue;
      if (a >= schd.ncore+schd.nact || b >= schd.ncore+schd.nact) continue;
      if (!(d.getocc(a) || d.getocc(b)) && a!=b) {
        Determinant di = d;
        di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);
        if(old_dets.find(di) == old_dets.end() && new_dets.find(di) == new_dets.end()) {
          new_dets.emplace(di);
        }
        Determinant detcpy(di);
        detcpy.flipAlphaBeta();
        if(old_dets.find(detcpy) == old_dets.end() && new_dets.find(detcpy) == new_dets.end()) {
          new_dets.emplace(detcpy);
        }
        //if (Determinant::Trev != 0) di.makeStandard();
      }
    } // heatbath integrals
  } // ij
  return;
} // end SHCIgetdeterminants::getDeterminantsVariational

vector<value_type> cdfci::getSubDets(value_type& d, hash_det& wfn, int nelec, bool sample) {
  auto det = d.first;
  int norbs = det.norbs;
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  vector<int> closed(nelec, 0);
  vector<int> open(nopen, 0);
  det.getOpenClosed(open, closed);
    
  vector<value_type> result;
  value_type val = std::make_pair(det, std::array<dcomplex, 2> {0.0, 0.0});
  result.push_back(val);

  //get all existing single excitationsget
  for (int ia=0; ia<nopen*nclosed; ia++) {
    int i=ia/nopen, a=ia%nopen;
    Determinant di = det;
    di.setocc(open[a], true);
    di.setocc(closed[i], false);
    //if (wfn.find(di) != wfn.end()) {
    value_type val = std::make_pair(di, std::array<dcomplex, 2> {0.0, 0.0});
    result.push_back(val);
  }

  //get all existing double excitations
  for(int ij=0; ij<nclosed*nclosed; ij++) {
    int i=ij/nclosed, j=ij%nclosed;
    if(i<=j) continue;
    int I = closed[i], J = closed[j];
    for (int kl=0;kl<nopen*nopen;kl++) {
      int k=kl/nopen;
      int l=kl%nopen;
      if(k<=l) continue;
      int K=open[k], L=open[l];
      int a = max(K,L), b = min(K,L);
      Determinant di = det;
      di.setocc(a, true);
      di.setocc(b, true);
      di.setocc(I, false);
      di.setocc(J, false);
      value_type val = std::make_pair(di, std::array<dcomplex, 2> {0.0, 0.0});
      result.push_back(val);
    }
  }
  return result;
}

value_type cdfci::CoordinatePickGcdGrad(vector<value_type> sub, double norm, bool real_part) {
  double max_abs_grad = 0;
  auto result = sub.begin();
  for(auto iter=sub.begin()+1; iter!=sub.end(); iter++) {
    auto x = iter->second[0];
    auto z = iter->second[1];
    auto grad_real = z.real()+x.real()*norm;
    auto grad_imag = z.imag()+x.imag()*norm;
    auto abs_grad = std::abs(grad_real+grad_imag);
    if (abs_grad >= max_abs_grad) {
      max_abs_grad = abs_grad;
      result = iter;
      if (std::abs(grad_real) > std::abs(grad_imag)) real_part = true;
      else real_part = false;
    }
  }
  return std::make_pair(result->first, result->second);
}

inline int sgn(double x) {
    if (x >= 0.0) return 1;
    else return -1;
}

double line_search(double p1, double q, double x) {
  double dx = 0.0;
  double p3 = p1/3;
  double q2 = q/2;
  double d = p3 * p3 * p3 + q2 * q2;
  double rt = 0;
  //std::cout << "p1: " << p1 << " q: " << q << std::endl;
  const double pi = atan(1.0) * 4;

  if (d >= 0) {
    auto qrtd = sqrt(d);
    rt = cbrt(-q2 + qrtd) + cbrt(-q2 - qrtd);
  }
  else {
    auto qrtd = sqrt(-d);
    if (q2 >= 0) {
      rt = 2 * sqrt(-p3) * cos((atan2(-qrtd, -q2) - 2*pi)/3.0);      
    }
    else {
      rt = 2 * sqrt(-p3) * cos(atan2(qrtd, -q2)/3.0);
    }
  }
  dx = rt - x;
// Newton iteration to improve accuracy
  auto dxn = dx - (dx*(dx*(dx + 3*x) + (3*x*x + p1)) + p1*x + q + x*x*x)
             / (dx*(3*dx + 6*x) + 3*x*x + p1);

  const double depsilon = 1e-12;
  const int max_iter = 10;
  int iter = 0;
  while (fabs((dxn - dx)/dx) > depsilon && iter < max_iter)
  {
      dx = dxn;
      dxn = dx - (dx*(dx*(dx + 3*x) + (3*x*x + p1)) + p1*x + q + x*x*x)
          / (dx*(3*dx + 6*x) + 3*x*x + p1);
      ++iter;
  }
  return dx;
}
dcomplex cdfci::CoordinateUpdate(value_type& det_picked, hash_det & wfn, pair<dcomplex,double>& ene, vector<double> E0, oneInt& I1, twoInt& I2, double& coreE, bool real_part) {
  dcomplex result;
  // coreE doesn't matter.
  double dx = 0.0;
  auto det = det_picked.first;
  auto x = det_picked.second[0];
  auto z = det_picked.second[1];
  auto xx = ene.second;
  size_t orbDiff;
  
  double dA = det.Energy(I1, I2, coreE);
  dA=-dA;
  xx = xx - norm(x);
  double x_re = x.real();
  double x_im = x.imag();
  double z_re = (z + dA * x).real();
  double z_im = (z + dA * x).imag();
  double p1_re = xx + x_im * x_im - dA;
  double q_re = z_re;  //
  double dx_re = line_search(p1_re, q_re, x_re);
  double p1_im = xx + x_re * x_re - dA;
  double q_im = z_im;
  double dx_im = line_search(p1_im, q_im, x_im);
  result = dcomplex(dx_re, dx_im);
  return result;
}

void cdfci::civectorUpdate(vector<value_type>& column, hash_det& wfn, dcomplex dx, pair<dcomplex, double>& ene, oneInt& I1, twoInt& I2, double& coreE, double thresh, bool sample) {
  auto deti = column[0].first;
  size_t orbDiff;
  auto entry=column.begin();
  double hij;
  hij = deti.Energy(I1, I2, coreE);
  auto x = wfn[deti][0];
  ene.first += conj(dx) * hij * dx 
            +  dx * hij * conj(x) 
            +  conj(dx) * hij * wfn[deti][0];
  ene.second += norm(wfn[deti][0]+dx)-norm(wfn[deti][0]);
  wfn[deti][0] += dx;
  wfn[deti][1] += hij*dx;
  column[0].second = wfn[deti];
  for (auto entry = column.begin()+1; entry!=column.end(); entry++) {
    auto detj = entry->first;
    dcomplex hij;
    auto iter = wfn.find(detj);
    hij = Hij(deti, detj, I1, I2, coreE, orbDiff);
    auto dz = dx*hij;
    if (iter != wfn.end()) {
      auto z = iter->second[0];
      ene.first += conj(z)*dz+conj(dz)*z;
      wfn[detj][1] += dz;
      entry->second=wfn[detj];
    }
    else {
      if (std::abs(dz) > thresh && sample) {
        wfn[detj]={0.0, dz};
        entry->second={0.0, dz};
      }
    }
  }
    
  return; 
}

cdfci::hash_det cdfci::precondition(pair<dcomplex, double>& ene, vector<Determinant>& dets, vector<MatrixXx>& ci, vector<double> energy, oneInt& I1, twoInt& I2, double& coreE, double thresh, bool sample) {
  auto norm = std::sqrt(std::abs(energy[0]-coreE));
  hash_det wfn;
  ene = std::make_pair(dcomplex(0.0, 0.0), 0.0);
  const int nelec = dets[0].Noccupied();
  dcomplex zero(0.0, 0.0);
  double coreEbkp = coreE;
  coreE = 0.0;
  std::array<dcomplex, 2> zeros{zero, zero};
  auto dets_size = dets.size();
  for (int i = 0; i < dets_size; i++) {
    wfn[dets[i]] = zeros;
  }
  for (int i = 0; i < dets_size; i++) {
    auto dx = ci[0](i,0)*norm;
    auto thisDet = std::make_pair(dets[i], wfn[dets[i]]);
    auto column = cdfci::getSubDets(thisDet, wfn, nelec, sample);
    cdfci::civectorUpdate(column, wfn, dx, ene, I1, I2, coreE, thresh, sample);
  }
  coreE = coreEbkp;
  return wfn;
}
void cdfci::cdfciSolver(hash_det& wfn, Determinant& hf, schedule& schd, pair<dcomplex, double>& ene, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, double& coreE, vector<double>& E0, int nelec, double thresh, bool sample) {
    // first to initialize hx for new determinants
    auto coreEbkp = coreE;
    coreE = 0.0;
    auto startofCalc = getTime();
    value_type thisDet =  std::make_pair(hf, wfn[hf]);
    
    double prev_ene;
    bool real_part = true;
    if (abs(ene.first) > 1e-10) {
      prev_ene = ene.first.real()/ene.second;
    }
    else{
      prev_ene=0.0;
    }

    auto num_iter = schd.cdfciIter;
    std::cout << "start optimization" << std::endl;
    std::cout << "energy " << prev_ene << " norm: " << ene.second << std::endl;
    if (std::abs(prev_ene) < 1e-10) {
      auto dx = -1.0*std::sqrt(std::abs(hf.Energy(I1, I2, coreE)));
      auto column = cdfci::getSubDets(thisDet, wfn, nelec, sample);
      cdfci::civectorUpdate(column, wfn, dx, ene, I1, I2, coreE, thresh, sample);
    }
    for(int k=0; k<num_iter; k++) {
      if (wfn.size() > schd.max_determinants) sample = false;
      auto dx = cdfci::CoordinateUpdate(thisDet, wfn, ene, E0, I1, I2, coreE, real_part);
      auto column = cdfci::getSubDets(thisDet, wfn, nelec, sample);
      cdfci::civectorUpdate(column, wfn, dx, ene, I1, I2, coreE, thresh, sample);
      if(k%schd.report_interval == 0) {
        auto curr_ene = ene.first.real()/ene.second;
        auto imag_ene = ene.first.imag()/ene.second;
        // iter, energy, time, variation space, dx
        std::cout << std::setw(10) << k <<  std::setw(20) <<std::setprecision(16) << defaultfloat << curr_ene+coreEbkp << std::setw(12) << setprecision(4) << imag_ene ;
        std::cout << std::setw(12) << std::setprecision(6) << dx << std::setw(10) << wfn.size() << std::setw(10) << std::setprecision(2) << getTime()-startofCalc << std::endl;
        if (std::abs(curr_ene-prev_ene)/schd.report_interval < schd.dE && k > 0) {
          break;
        }
        prev_ene = curr_ene;
      }
      thisDet = cdfci::CoordinatePickGcdGrad(column, ene.second, real_part);
    }
  coreE=coreEbkp;
  auto factor = std::sqrt(std::abs(ene.second));
  for(auto iter=wfn.begin(); iter!=wfn.end(); ++iter) {
    if(std::abs(iter->second[0]) > 0.1)
      std::cout << "Det: " << iter->first << " coeff : " << iter->second[0]/factor << std::endl;
  }
  return;
}

void cdfci::cyclicSolver(hash_det& wfn, vector<Determinant>& Dets, schedule& schd, pair<dcomplex, double>& ene, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, double& coreE, vector<double>& E0, int nelec, double thresh, bool sample) {
    // first to initialize hx for new determinants
    auto coreEbkp = coreE;
    //auto coreEbkp=0.0;
    coreE = 0.0;
    auto startofCalc = getTime();

    double prev_ene;
    if (abs(ene.first) > 1e-10) {
      prev_ene = ene.first.real()/ene.second;
    }
    else{
      prev_ene=0.0;
    }

    auto num_iter = schd.cdfciIter;
    std::cout << "start optimization" << std::endl;
    int dets_size = Dets.size();
    bool real_part = true;
    for(int k=0; k<num_iter; k++) {
        value_type thisDet = std::make_pair(Dets[k%dets_size], wfn[Dets[k%dets_size]]);
        auto dx = cdfci::CoordinateUpdate(thisDet, wfn, ene, E0, I1, I2, coreE, true);
        auto column = cdfci::getSubDets(thisDet, wfn, nelec, sample);
        cdfci::civectorUpdate(column, wfn, dx, ene, I1, I2, coreE, thresh, sample);
        dx = cdfci::CoordinateUpdate(thisDet, wfn, ene, E0, I1, I2, coreE, false);
        cdfci::civectorUpdate(column, wfn, dx, ene, I1, I2, coreE, thresh, sample);
      if(k%schd.report_interval == 0) {
        if (wfn.size() > schd.max_determinants) sample = false;
        auto curr_ene = ene.first.real()/ene.second;
        auto imag_ene = ene.first.imag()/ene.second;
        // iter, energy, time, variation space, dx
        std::cout << std::setw(10) << k <<  std::setw(20) <<std::setprecision(16) << curr_ene+coreEbkp << std::setw(12) << setprecision(4) << imag_ene << " " << thisDet.first;
        std::cout << std::setw(12) << std::setprecision(6) << dx << std::setw(10) << wfn.size() << std::setw(10) << std::setprecision(2) << getTime()-startofCalc << std::endl;
        if (std::abs(curr_ene-prev_ene) < schd.dE) {
          break;
        }
        prev_ene = curr_ene;
      }
    }
  coreE=coreEbkp;
  for(auto iter=wfn.begin(); iter!=wfn.end(); ++iter) {
    if(std::norm(iter->second[0]) > 0.1)
      std::cout << "Det: " << iter->first << " coeff : " << norm(iter->second[0])/ene.second << std::endl;
  }
  return;
}

// this returns the numerator change of the rayleigh quotient.
void civectorUpdateNoSample(pair<double, double>& ene, vector<int>& column, dcomplex dx, Determinant* dets, vector<dcomplex>& x_vector, vector<dcomplex>& z_vector, DetToIndex& det_to_index, oneInt& I1, twoInt& I2, double& coreE) {
  auto deti = dets[column[0]];
  size_t orbDiff;
  double hij;
  hij = deti.Energy(I1, I2, coreE);
  auto dz = hij * dx;
  z_vector[column[0]] += dz;
  auto z = z_vector[column[0]];
  auto x = x_vector[column[0]];
  x_vector[column[0]] += dx;
  ene.first += (conj(dx) * hij * dx 
            +  dx * hij * conj(x) 
            +  conj(dx) * hij * x).real();
  ene.second += norm(x + dx) - norm(x);
  double local_norm;
  double global_norm;
  int num_iter = column.size();
  #pragma omp parallel private(local_norm, dz, x) shared(global_norm)
  {
    local_norm = 0.0;
    global_norm = 0.0;
  #pragma omp for
  for (int i = 1; i < num_iter; i++) {
    auto detj = dets[column[i]];
    dcomplex hij = Hij(deti, detj, I1, I2, coreE, orbDiff);
    dz = dx * hij;
    x = x_vector[column[i]];
    z_vector[column[i]] += dz;
    local_norm += (conj(x)*dz+conj(dz)*x).real();
  }
  #pragma omp critical
  global_norm += local_norm;
  #pragma omp barrier
  }
  ene.first += global_norm;
  return;
}

dcomplex CoordinateUpdate(Determinant& det, dcomplex x, double z_re, double z_im, double xx, oneInt& I1, twoInt& I2, double& coreE) {
  dcomplex result;
  double dx = 0.0;
  size_t orbDiff;
  
  double dA = -det.Energy(I1, I2, coreE);
  xx = xx - norm(x);
  double x_re = x.real(), x_im = x.imag();
  double q_re = z_re + dA * x_re; 
  double q_im = z_im + dA * x_im;
  double p1_re = xx + x_im * x_im  - dA;
  double dx_re = line_search(p1_re, q_re, x_re);
  double p1_im = xx + x_re * x_re - dA;
  double dx_im = line_search(p1_im, q_im, x_im);
  if (abs(dx_re) > abs(dx_im)) return dcomplex(dx_re, 0.0);
  else return dcomplex(0.0, dx_im);
  return result;
}

dcomplex CoordinateUpdateIthRoot(Determinant& det, vector<dcomplex>& x, vector<double>& z_re, vector<double>&z_im, vector<vector<double>>& xx_re, vector<vector<double>>& xx_im, int& iroot, oneInt& I1, twoInt& I2, double& coreE) {
  dcomplex result;
  double dx = 0.0;
  size_t orbDiff;
  double dA = -det.Energy(I1, I2, coreE);
  const int nroots = x.size();
  double x_re = x[iroot].real(), x_im = x[iroot].imag();
  double p1_re = xx_re[iroot][iroot] - x_re * x_re - dA;
  double p1_im = xx_re[iroot][iroot] - x_im * x_im - dA;
  double q_re = z_re[iroot] + dA * x_re;
  double q_im = z_im[iroot] + dA * x_im;
  
  for(int jroot = 0; jroot < nroots; jroot++) {
    if (jroot != iroot) {
    double xj_re = x[jroot].real(), xj_im = x[jroot].imag();
    auto xj_norm = xj_re*xj_re + xj_im*xj_im;
    p1_re += xj_norm;
    p1_im += xj_norm;
    q_re += xj_re * xx_re[iroot][jroot] + xj_im * xx_im[iroot][jroot] - xj_norm * x_re;
    q_im += xj_im * xx_re[iroot][jroot] - xj_re * xx_im[iroot][jroot] - xj_norm * x_im;
    }
  }

  double dx_re = line_search(p1_re, q_re, x_re);
  double dx_im = line_search(p1_im, q_im, x_im);
  if (abs(dx_re) > abs(dx_im)) return dcomplex(dx_re, 0.0);
  else return dcomplex(0.0, dx_im);
}

void getSubDetsNoSample(vector<Determinant>& dets, vector<int>& column, DetToIndex& det_to_index, int this_index, int nelec) {
  auto det = dets[this_index];

  int norbs = det.norbs;
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  vector<int> closed(nelec, 0);
  vector<int> open(nopen, 0);
  det.getOpenClosed(open, closed);
  vector<int> result(1+nopen*nclosed+nopen*nopen*nclosed*nclosed, -1);
 
  result[0] = this_index;

  for (int ia=0; ia<nopen*nclosed; ia++) {
    int i=ia/nopen, a=ia%nopen;
    Determinant di = det;
    di.setocc(open[a], true);
    di.setocc(closed[i], false);
    auto iter = det_to_index.find(di);
    if (iter != det_to_index.end()) {
      result[ia+1] = iter->second;
    }
  }

  //get all existing double excitations
  for(int ij=0; ij<nclosed*nclosed; ij++) {
    int i=ij/nclosed, j=ij%nclosed;
    if(i<=j) continue;
    int I = closed[i], J = closed[j];
    for (int kl=0;kl<nopen*nopen;kl++) {
      int k=kl/nopen;
      int l=kl%nopen;
      if(k<=l) continue;
      int K=open[k], L=open[l];
      int a = max(K,L), b = min(K,L);
      Determinant di = det;
      di.setocc(a, true);
      di.setocc(b, true);
      di.setocc(I, false);
      di.setocc(J, false);
      auto iter = det_to_index.find(di);
      if (iter != det_to_index.end()) {
        result[1+nopen*nclosed+ij*nopen*nopen+kl] = iter->second;
      }
    }
  }

  column.clear();
  column.push_back(result[0]);
  //#pragma omp parallel for
  for (int i = 1; i < 1+nopen*nclosed+nopen*nopen*nclosed*nclosed; i++) {
    if (result[i]>=0) 
    {
      column.push_back(result[i]);
    }
  }
  return;
}

int CoordinatePickGcdGradOmp(vector<int>& column, vector<dcomplex>& x_vector, vector<dcomplex>& z_vector, vector<pair<double, double>>& ene) {
  double max_abs_grad = 0.0;
  int result = -1;
  // only optimize ground state
  double norm = ene[0].second;
  #pragma omp parallel for default(none) shared(x_vector, z_vector, column, max_abs_grad, result, norm)
  for(int i = 1; i < column.size(); i++) {
    auto x = x_vector[column[i]];
    auto z = z_vector[column[i]];
    auto grad_real = z.real()+x.real()*norm;
    auto grad_imag = z.imag()+x.imag()*norm;
    auto abs_grad = std::abs(grad_real+grad_imag);
    if (abs_grad > max_abs_grad) {
      max_abs_grad = abs_grad;
      result = column[i];
    }
  }
  return result;
}

vector<pair<double, double>> precondition(vector<dcomplex>& x_vector, vector<double>& z_vec_re, vector<double>& z_vec_im, vector<MatrixXx>& ci, DetToIndex& det_to_index, vector<Determinant>& dets, vector<double>& E0, oneInt& I1, twoInt& I2, double coreE) {
  int nelec = dets[0].Noccupied();
  int nroots = ci.size();
  vector<pair<double, double>> result(nroots, make_pair<double, double>(0.0, 0.0));
  vector<int> column;
  
  for (int iroot = 0; iroot < nroots; iroot++) {
    int x_size = ci[iroot].rows();
    double norm = sqrt(abs(E0[iroot]-coreE));
    auto result_iroot = pair<double, double>(0.0, 0.0);
    for (int i = 0; i < x_size; i++) {
      auto dx = ci[iroot](i, 0) * norm;
      result_iroot.second += std::norm(dx);
      x_vector[i*nroots+iroot] = dx;
    }
    double xz = 0.0;
    //#pragma omp declare reduction(complex_plus : dcomplex : std::plus<dcomplex>())
    //#pragma omp parallel for private(column) reduction(complex_plus : xz)
    for (int i = 0; i < x_size; i++) {
      getSubDetsNoSample(dets, column, det_to_index, i, nelec);

      auto column_size = column.size();
      auto deti = dets[i];
      auto hij = deti.Energy(I1, I2, coreE);
      auto xi = x_vector[i*nroots+iroot];
      xz += std::norm(xi) * hij;
      z_vec_re[i*nroots+iroot] += (hij*xi).real();
      z_vec_im[i*nroots+iroot] += (hij*xi).imag();
      for (int entry = 1; entry < column_size; entry++) {
        auto j = column[entry];
        auto detj = dets[j];
        auto xj = x_vector[j*nroots+iroot];
        size_t orbDiff;
        auto hij = Hij(deti, detj, I1, I2, coreE, orbDiff);
        xz += (conj(xj) * hij * xi).real();
        #pragma omp atomic
        z_vec_re[j*nroots+iroot] += (hij*xi).real();
        #pragma omp atomic
        z_vec_im[j*nroots+iroot] += (hij*xi).imag();
      }
    }
    result_iroot.first = xz;
    result[iroot] = result_iroot;
    auto residual = cdfci::compute_residual(x_vector, z_vec_re, z_vec_im, result, iroot);
    cout << iroot << " " <<  residual << endl;
  }
  return result;
}

double cdfci::compute_residual(vector<dcomplex>& x, vector<double>& zreal, vector<double>& zimag, vector<pair<double, double>>& ene, int& iroot) {
  auto energy = ene[iroot].first / ene[iroot].second;
  double residual = 0.0;
  const int size = x.size();
  const int nroots = ene.size();
  #pragma omp parallel reduction(+:residual)
  for (int i = iroot; i < size; i+=nroots) {
    auto tmp_re = zreal[i] - energy * x[i].real();
    auto tmp_im = zimag[i] - energy * x[i].imag();
    residual += tmp_re*tmp_re + tmp_im*tmp_im;
  }
  return residual;
}

double get_energy(pair<double, double> energy) {
  return energy.first / energy.second;
}
void cdfci::solve(schedule& schd, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, double& coreE, vector<double>& E0, vector<MatrixXx>& ci, vector<Determinant>& dets) {
  DetToIndex det_to_index;
  robin_hood::unordered_set<Determinant> old_dets;
  robin_hood::unordered_set<Determinant> new_dets;
  int iter;
  bool converged;
  int thread_id;
  int thread_num;
  #pragma omp parallel
  {
    thread_num = omp_get_num_threads();
    cout << thread_num << endl;
  }
  double coreEbkp = coreE;
  coreE = 0.0;

  if (schd.restart) {
    char file[5000];
    sprintf(file, "%s/%d-variational.bkp", schd.prefix[0].c_str(), commrank);
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);

    load >> iter >> dets;
    ci.resize(1, MatrixXx(dets.size(), 1));

    load >> ci;
    load >> E0;
    load >> converged;
    pout << "Load converged: " << converged << endl;
  }

  int start_index = 0;//ci[0].rows();
  int dets_size = dets.size();

  for (int i = 0; i < dets_size; i++) {
    det_to_index[dets[i]] = i;
    old_dets.emplace(dets[i]);
  }

  const double epsilon1 = schd.epsilon1[schd.cdfci_on];
  const int nelec = dets[0].Noccupied();
  const dcomplex zero = 0.0;

  for (int i = 0; i < dets_size; i++) {
    double cmax = 0.0;
    for(int iroot = 0; iroot < ci.size(); iroot++) {
      cmax += norm(ci[iroot](i,0));
    }
    cmax = sqrt(cmax);
    cdfci::getDeterminantsVariational(dets[i], epsilon1/cmax, cmax, zero, I1, I2, I2HB, irrep, coreE, E0[0], old_dets, new_dets, schd, 0, nelec);
    if (i%10000 == 0) {
      cout << "curr iter " << i << " new dets size " << 
new_dets.size() << endl;
    }
  }
  int new_dets_size = new_dets.size();
  for (auto new_det : new_dets) {
    dets.push_back(new_det);
  }
  dets_size = dets.size();
  robin_hood::unordered_set<Determinant>().swap(old_dets);
  robin_hood::unordered_set<Determinant>().swap(new_dets);
  old_dets.clear();
  new_dets.clear();
  cout << "build det to index" << endl;
  for (int i = 0; i < dets_size; i++) {
    det_to_index[dets[i]] = i;
    if (i % 10000 == 0) {
      cout << i << "dets constructed" << endl;
    }
  }
  cout << det_to_index.size() << " determinants optimized by cdfci" << endl;
  // ene stores the rayleigh quotient quantities.
  int nroots = schd.nroots;
  vector<pair<double, double>> ene(nroots, make_pair(0.0, 0.0));
  vector<dcomplex> x_vector(dets_size * nroots, zero);
  vector<double> z_vec_re(dets_size * nroots, 0.0);
  vector<double> z_vec_im(dets_size * nroots, 0.0);
  auto start_time = getTime();
  ene = precondition(x_vector, z_vec_re, z_vec_im, ci, det_to_index, dets, E0, I1, I2, coreE);
  vector<vector<double>> xx_re(nroots, vector<double>(nroots, 0.0));
  vector<vector<double>> xx_im(nroots, vector<double>(nroots, 0.0));

  for (int iroot = 0; iroot<nroots; iroot++) 
    xx_re[iroot][iroot] = ene[iroot].second;
  cout << get_energy(ene[0]) + coreEbkp << endl;
  auto num_iter = schd.cdfciIter;
  vector<int> this_det_idx(thread_num, 0);
  vector<dcomplex> dxs(thread_num, 0.0);
  vector<int> column;

  #pragma omp parallel
  {
    thread_id = omp_get_thread_num();
    this_det_idx[thread_id] = start_index + thread_id;
  }

  cout << "start to optimize" << endl;
  for (int iroot = 0; iroot < nroots; iroot++) {
    auto prev_ene = get_energy(ene[iroot]);
    auto start_time = getTime();
    cout << iroot << endl;
    for (int iter = 0; iter*thread_num <= num_iter; iter++) {

      // initialize dx on each thread
      {
        vector<vector<dcomplex>> x;
        vector<vector<double>> z_re;
        vector<vector<double>> z_im;
        size_t orbDiff;
        for (int thread = 0; thread < thread_num; thread++) {
          auto idx = this_det_idx[thread];
          auto x_first = x_vector.begin()+nroots*idx;
          auto x_last = x_first + nroots;
          auto zre_first = z_vec_re.begin() + nroots*idx;
          auto zre_last = zre_first +nroots;
          auto zim_first = z_vec_im.begin() + nroots*idx;
          auto zim_last = zim_first + nroots;
          x.push_back(vector<dcomplex>(x_first, x_last));
          z_re.push_back(vector<double>(zre_first, zre_last));
          z_im.push_back(vector<double>(zim_first, zim_last));
        }
        for (int thread = 0; thread < thread_num; thread++) {
          auto deti_idx = this_det_idx[thread];
          auto i_idx = deti_idx*nroots+iroot;
          dxs[thread] = CoordinateUpdateIthRoot(dets[deti_idx], x[thread], z_re[thread], z_im[thread], xx_re, xx_im, iroot, I1, I2, coreE);
          auto dx = dxs[thread];
          double hij = dets[deti_idx].Energy(I1, I2, coreE);
          auto xi = x_vector[i_idx];
          ene[iroot].first += (std::norm(dx) * hij
                            +  2. * dx.real() * hij * xi.real()
                            +  2. * dx.imag() * hij * xi.imag());
          ene[iroot].second += std::norm(dx+xi) - std::norm(xi);
          x_vector[i_idx] += dx;
          z_vec_re[i_idx] += hij*dx.real();
          z_vec_im[i_idx] += hij*dx.imag();
          // update x^\dagger_i x_j matrix.
          // first index be the conjugate one.
          xx_re[iroot][iroot] += norm(dx);
          for (int jroot = 0; jroot < nroots; jroot++) {
            //<x_i^\dagger| x_j>
            dcomplex dij = conj(dx) * x[thread][jroot];
            double dij_re = dij.real();
            double dij_im = dij.imag();
            xx_re[iroot][jroot] += dij_re;
            xx_re[jroot][iroot] += dij_re;
            xx_im[iroot][jroot] += dij_im; 
            xx_im[jroot][iroot] -= dij_im;
          }

          for (int thread_j = thread+1; thread_j < thread_num; thread_j++) {
            auto detj_idx = this_det_idx[thread_j];
            if (dets[deti_idx].ExcitationDistance(dets[detj_idx]) > 2 || deti_idx==detj_idx) continue;
            else {
              auto hij = Hij(dets[deti_idx], dets[detj_idx], I1, I2, coreE, orbDiff);
              z_re[thread_j][iroot] += (dx * hij).real();
              z_im[thread_j][iroot] += (dx * hij).imag();
            }
          }
        }

        // this step is because, although we are updating all the walkers simultaneously
        // what in fact want to mimic the result of sequential update.
        // but, we are updating all the x_vectors at the very beginning.
        // the "later" x_vectors are updated earlier than expected, and will have an
        // influence on the xz/ene.first term. So this influence needs to be deducted.
        // xz += x*conj(dz) + conj(x)*dz. So dx*conj(dz)+conj(dx)*dz should be deducted. 
        // Only terms with non vanishing dx has a contribution.
        // which means that, only the dets with a to be updated x[j] will be affected.,
        // xz -= dx[j]*conj(dz[i]) + conj(dx[j])*dz[i] dz[i] = hij*dx[i], with j > i. ,
        // Because only when j > i, the future happening update is affecting the past.
        for (int thread_i = 0; thread_i < thread_num; thread_i++) {
          auto deti_idx = this_det_idx[thread_i];
          auto deti = dets[deti_idx];
          auto dxi = dxs[thread_i];
          for(int thread_j = thread_i + 1; thread_j < thread_num; thread_j++) {
            auto detj_idx = this_det_idx[thread_j];
            auto detj = dets[detj_idx];
            if (deti.ExcitationDistance(detj) > 2 || deti_idx == detj_idx) continue;
            else {
              auto dxj = dxs[thread_j];
              auto hij = Hij(deti, detj, I1, I2, coreE, orbDiff);
              auto dzi = hij * dxi;
              ene[iroot].first -= 2.*(dxj.real()*dzi.real()+dxj.imag()*dzi.imag());
            }
          }
        }
      }

      const int norbs = dets[0].norbs;
      const int nclosed = dets[0].Noccupied();
      const int nopen = norbs - nclosed;

      #pragma omp parallel private(column)
      {
        int thread_id = omp_get_thread_num();
        int deti_idx = this_det_idx[thread_id];
        int i_idx = deti_idx*nroots+iroot;
        auto deti = dets[deti_idx];
        vector<int> closed(nelec, 0);
        vector<int> open(nopen, 0);
        deti.getOpenClosed(open, closed);
        const auto xx = ene[iroot].second;
        double max_abs_grad = 0.0;
        int selected_det = deti_idx+thread_num;

        auto dx = dxs[thread_id];
        bool real_part;
        if (abs(dx.real()) < 1e-20) real_part = false;
        else real_part = true;
        auto det_energy = deti.Energy(I1, I2, coreE);
        auto dz = dx * det_energy;
        auto x = x_vector[i_idx];
        double xz = 0.0;

        for (int ia = 0; ia < nopen * nclosed; ia++) {
          int i = ia / nopen, a = ia % nopen;
          auto detj = deti;
          detj.setocc(open[a], true);
          detj.setocc(closed[i], false);
          auto iter = det_to_index.find(detj);
          if (iter != det_to_index.end()) {
            auto detj_idx = iter->second;
            auto j_idx = detj_idx * nroots+iroot;
            size_t orbDiff;
            auto hij = Hij(deti, detj, I1, I2, coreE, orbDiff);
            auto dz = dx * hij;
            auto x = x_vector[j_idx];
            #pragma omp atomic
            z_vec_re[j_idx] += dz.real();
            #pragma omp atomic
            z_vec_im[j_idx] += dz.imag();
            xz += 2.*(x.imag()*dz.imag()+x.real()*dz.real());
            if (j_idx % thread_num == thread_id) {
              auto grad_real = z_vec_re[j_idx] + x.real() * xx;
              auto grad_imag = z_vec_im[j_idx] + x.imag() * xx;
              for(int jroot=0; jroot<nroots; jroot++) {
                if(jroot != iroot) {
                  auto xj_jroot = x_vector[detj_idx*nroots+jroot];
                  auto xj_jroot_re = xj_jroot.real();
                  auto xj_jroot_im = xj_jroot.imag();
                  grad_real += xj_jroot_re*xx_re[iroot][jroot]
                             + xj_jroot_im*xx_im[iroot][jroot];
                  grad_imag += xj_jroot_im*xx_re[iroot][jroot]
                             - xj_jroot_re*xx_im[iroot][jroot];
                }
              }
              auto abs_grad = abs(grad_real + grad_imag);
              if (abs_grad > max_abs_grad) {
                max_abs_grad = abs_grad;
                selected_det = detj_idx;
              } 
            }
          }
        } // single excitation ends.

        for(int ij = 0; ij < nclosed * nclosed; ij++) {
          int i = ij / nclosed, j = ij % nclosed;
          if(i <= j) continue;
          int I = closed[i], J = closed[j];
          for (int kl = 0; kl < nopen * nopen; kl++) {
            int k = kl / nopen;
            int l = kl % nopen;
            if (k <= l) continue;
            int K=open[k], L=open[l];
            int a = max(K,L), b = min(K,L);
            auto detj = deti;
            detj.setocc(a, true);
            detj.setocc(b, true);
            detj.setocc(I, false);
            detj.setocc(J, false);
            auto iter = det_to_index.find(detj);
            if (iter != det_to_index.end()) {
              auto detj_idx = iter->second;
              auto j_idx = detj_idx * nroots+iroot;
              size_t orbDiff;
              auto hij = Hij(deti, detj, I1, I2, coreE, orbDiff);
              auto dz = dx * hij;
              auto x = x_vector[j_idx];
              #pragma omp atomic
              z_vec_re[j_idx] += dz.real();
              #pragma omp atomic
              z_vec_im[j_idx] += dz.imag();
              xz += 2.*(x.imag()*dz.imag()+x.real()*dz.real());
              if (j_idx % thread_num == thread_id) {
                auto grad_real = z_vec_re[j_idx] + x.real() * xx;
                auto grad_imag = z_vec_im[j_idx] + x.imag() * xx;
                for(int jroot=0; jroot<nroots; jroot++) {
                  if(jroot != iroot) {
                    auto xj_jroot = x_vector[detj_idx*nroots+jroot];
                    auto xj_jroot_re = xj_jroot.real();
                    auto xj_jroot_im = xj_jroot.imag();
                    grad_real += xj_jroot_re*xx_re[iroot][jroot]
                               + xj_jroot_im*xx_im[iroot][jroot];
                    grad_imag += xj_jroot_im*xx_re[iroot][jroot]
                               - xj_jroot_re*xx_im[iroot][jroot];
                  }
                }
                auto abs_grad = abs(grad_real + grad_imag);
                if (abs_grad > max_abs_grad) {
                  max_abs_grad = abs_grad;
                  selected_det = detj_idx;
                } 
              }
            }
          }
        }
        this_det_idx[thread_id] = selected_det;
        #pragma omp atomic
        ene[iroot].first += xz;
      }
      #pragma omp barrier
      
      // now some logical codes, print out information and decide when to exit etc.
      if (iter*thread_num%schd.report_interval < thread_num || iter == num_iter) {
        auto curr_ene = get_energy(ene[iroot]);
        auto residual = cdfci::compute_residual(x_vector, z_vec_re, z_vec_im, ene, iroot);
        cout << setw(10) << iter * thread_num << setw(20) <<std::setprecision(14) << curr_ene+coreEbkp << setw(20) <<std::setprecision(14) << prev_ene+coreEbkp;
        cout << std::setw(12) << std::setprecision(4) << scientific << getTime()-start_time << defaultfloat;
        cout << std::setw(12) << std::setprecision(4) << scientific << residual << defaultfloat << endl;
        if (residual < schd.cdfciTol) {
          break;
        }
        prev_ene = curr_ene;
      }
    }
  }
  coreE = coreEbkp;
  return;
}
