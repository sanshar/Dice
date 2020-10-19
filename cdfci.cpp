#include "cdfci.h"
#include "input.h"
#include "math.h"
#include "SHCIbasics.h"
#include <iostream>
#include <unordered_map>
#include <map>
#include <tuple>
#include <vector>
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
using namespace boost;
using namespace cdfci;
using StitchDEH = SHCISortMpiUtils::StitchDEH;
using cdfci::value_type;
typedef unordered_map<Determinant, array<dcomplex, 2>> hash_det;

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
      //std::cout << iter->first << " abs grad : " << scientific << std::setprecision(2) << setw(12) << grad_real << setw(12) << grad_imag << " " << real_part << std::endl;
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
  //std::cout << "selected det: " << deti << std::endl;
  size_t orbDiff;
  //std::cout << ene.first << " " << ene.second << std::endl;
  auto entry=column.begin();
  double hij;
  hij = deti.Energy(I1, I2, coreE);
  auto x = wfn[deti][0];
  ene.first += conj(dx) * hij * dx 
            +  dx * hij * conj(x) 
            +  conj(dx) * hij * wfn[deti][0];
  ene.second += norm(wfn[deti][0]+dx)-norm(wfn[deti][0]);
  //std::cout << wfn[detj][1] << " " << wfn[detj][0] << " " << ene.first << std::endl;
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
