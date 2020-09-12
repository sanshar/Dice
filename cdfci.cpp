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
    //}
    /*else {
       if (sample == true) {
        value_type val = std::make_pair(di, std::array<double, 2> {0.0, 0.0});
        result.push_back(val); 
      }
    }*/
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
      //if (wfn.find(di)!=wfn.end()){
      value_type val = std::make_pair(di, std::array<dcomplex, 2> {0.0, 0.0});
      result.push_back(val);
      //}
      //else {
      //  if (sample == true) {
      //    value_type val = std::make_pair(di, std::array<double, 2> {0.0, 0.0});
      //    result.push_back(val); 
      //  }
      //}
    }
  }
  return result;
}

value_type cdfci::CoordinatePickGcdGrad(vector<value_type> sub, double norm) {
  double max_abs_grad = 0;
  auto result = sub.begin();
  for(auto iter=sub.begin()+1; iter!=sub.end(); iter++) {
    auto x = iter->second[0];
    auto z = iter->second[1];
    double abs_grad = std::abs(x*norm+z);
    if (abs_grad >= max_abs_grad) {
      max_abs_grad = abs_grad;
      result = iter;
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
dcomplex cdfci::CoordinateUpdate(value_type& det_picked, hash_det & wfn, pair<dcomplex,double>& ene, vector<double> E0, oneInt& I1, twoInt& I2, double& coreE) {
  // coreE doesn't matter.
  double dx = 0.0;
  auto det = det_picked.first;
  auto x = det_picked.second[0];
  auto z = det_picked.second[1];
  auto xx = ene.second;
  //std::cout << x << " z:" << z << " xx:" << xx << std::endl; 
  size_t orbDiff;
  
  double dA = det.Energy(I1, I2, coreE);
  dA=-dA;
  double x_re = x.real();
  double x_im = x.imag();
  double z_re = z.real();
  double z_im = z.imag();
  double p1_re = xx - x_re * x_re - dA;
  double p1_im = xx - x_im * x_im;
  double q_re = z_re + dA * x_re;  //
  double q_im = z_im;
  double dx_re = line_search(p1_re, q_re, x_re);
  double dx_im = line_search(p1_im, q_im, x_im);
  return dcomplex(dx_re, dx_im);
  //if (abs(x)+abs(xx)+abs(z)<1e-100) return sqrt(abs(dA));
  //else return -0.05*(x*xx+z);
  //std::cout << "p1: " << p1 << " q: " << q << std::endl;
  //const double pi = atan(1.0) * 4;
  //TODO: need to think about complex value line search
  /*if (d >= 0) {
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
  }*/
  //return dx;
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
    /*if (detj == deti) {
      hij = deti.Energy(I1, I2, coreE);
      ene.first += hij * dx * dx+2*dx*hij*wfn[detj][0];
      ene.second += dx*dx + 2* wfn[detj][0]*dx;
      //std::cout << wfn[detj][1] << " " << wfn[detj][0] << " " << ene.first << std::endl;
      wfn[detj][0] += dx;
      wfn[detj][1] += hij*dx;
    }*/
    //else {
      hij = Hij(deti, detj, I1, I2, coreE, orbDiff);
      auto dz = dx*hij;
      /*if (sample == false) {
        ene.first += 2*iter->second[0]*dz;
        wfn[detj][1] += dz;
      }*/
      //else {
        if (iter != wfn.end()) {
          auto z = iter->second[0];
          ene.first += conj(z)*dz+conj(dz)*z;
          wfn[detj][1] += dz;
          entry->second=wfn[detj];
        }
        else {
          if (std::abs(dz) > thresh) {
            wfn[detj]={0.0, dz};
            entry->second={0.0, dz};
          }
        }
      //}
    //}
    
    //std::cout << "det:" << detj << " civec:" << wfn[detj][0] << " hx:" << wfn[detj][1] << std::endl;
  }
    
  return; 
}
 
void cdfci::cdfciSolver(hash_det& wfn, Determinant& hf, schedule& schd, pair<dcomplex, double>& ene, oneInt& I1, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, double& coreE, vector<double>& E0, int nelec, double thresh, bool sample) {
    // first to initialize hx for new determinants
    auto coreEbkp = coreE;
    //auto coreEbkp=0.0;
    coreE = 0.0;
    auto startofCalc = getTime();
    value_type thisDet =  std::make_pair(hf, wfn[hf]);
    //auto dx = -wfn[hf][0];
    //auto column=cdfci::getSubDets(thisDet, wfn, nelec, sample);
    //cdfci::civectorUpdate(column, wfn, dx, ene, I1, I2, coreE, thresh, sample);
    // assume non zero energy means start from a converged reference result
    /*if (ene.first < 0.0) {
      //auto new_dets = sampleExtraEntry(wfn, nelec);
      // add extra entries according to current wfn
      // keep old wfn, and have a new wfn that stores the updated wfn.
      hash_det new_dets = wfn; 
      //double epsilon = schd.z_threshold;
      for (auto det : wfn) {
        value_type det_val = std::make_pair(det.first, det.second);
        auto column = cdfci::getSubDets(det_val, new_dets, nelec, true);
        std::cout << "column size " << column.size() << std::endl;
        size_t orbDiff;
        auto xj = det.second[0];
        for(auto entry : column) {
          auto detj = entry.first;
          double z = 0.0;
          if (new_dets.find(detj) == new_dets.end()) {
            double hij = Hij(det.first, detj, I1, I2, coreE, orbDiff);
            z=xj*hij;
            if (fabs(z) > thresh) {
            // if z is above threshold, update z exactly.
              value_type detj_val = std::make_pair(detj, std::array<double, 2> {0.0, z});
              auto columnj = cdfci::getSubDets(detj_val, wfn, nelec, false);
              for(auto entry_j : columnj) {
                //std::cout << "columnj size " << columnj.size() << std::endl;
                if (entry_j.first == detj) continue;
                else {
                  double hij = Hij(entry_j.first, detj, I1, I2, coreE, orbDiff);
                  z+= hij*entry_j.second[0];
                }
              }
              new_dets[detj] = {0.0, z};
            }
          }
          else continue;
        }
        if (new_dets.size()-wfn.size() > 10000) {
          break;
        }
      }
      std::cout << "new dets size " << new_dets.size() << std::endl;
      wfn = new_dets;
      new_dets.clear();
      std::cout << "z space size " << wfn.size() << std::endl;
    }*/
    double prev_ene;
    if (abs(ene.first) > 1e-10)
      prev_ene = ene.first.real()/ene.second;
    else
      prev_ene=0.0;
    //prev_ene=0.0;
    //else prev_ene = ene.first/ene.second;
    /*if (ene.second > 1e-10) {
      //if start from a converged eigensystem perturb the system by a bit.
        auto column = cdfci::getSubDets(thisDet, wfn, nelec, sample);
        cdfci::civectorUpdate(column, wfn, 1e-2*wfn[hf][0], ene, I1, I2, coreE, thresh, sample);
        thisDet = cdfci::CoordinatePickGcdGrad(column, ene.second);
    }*/
    auto num_iter = schd.cdfciIter;
    std::cout << "start optimization" << std::endl;
    for(int k=0; k<num_iter; k++) {
        auto dx = cdfci::CoordinateUpdate(thisDet, wfn, ene, E0, I1, I2, coreE);
        auto column = cdfci::getSubDets(thisDet, wfn, nelec, sample);
        cdfci::civectorUpdate(column, wfn, dx, ene, I1, I2, coreE, thresh, sample);
        //sub = cdfci::getSubDets(thisDet, wfn, nelec);
        thisDet = cdfci::CoordinatePickGcdGrad(column, ene.second);
      if(k%schd.report_interval == 0) {
        if (wfn.size() > schd.max_determinants) sample = false;
        auto curr_ene = ene.first.real()/ene.second;
        auto imag_ene = ene.first.imag()/ene.second;
        // iter, energy, time, variation space, dx
        std::cout << std::setw(10) << k <<  std::setw(16) <<std::setprecision(12) << curr_ene+coreEbkp << std::setw(10) << setprecision(6) << imag_ene ;
        std::cout << std::setw(12) << std::setprecision(6) << dx << std::setw(10) << wfn.size() << std::setw(10) << std::setprecision(2) << getTime()-startofCalc << std::endl;
        if (std::abs(curr_ene-prev_ene) < schd.dE) {
          break;
        }
        prev_ene = curr_ene;
      }
    }
  coreE=coreEbkp;
  auto factor = sqrt(abs(ene.second));
  for(auto iter=wfn.begin(); iter!=wfn.end(); ++iter) {
      std::cout << "Det: " << iter->first << " coeff : " << iter->second[0]/factor << std::endl;
  }
  return;
}
