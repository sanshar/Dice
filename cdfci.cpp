#include "cdfci.h"
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

using namespace std;
using namespace Eigen;
using namespace boost;
using StitchDEH = SHCISortMpiUtils::StitchDEH;
using cdfci::value_type;
typedef unordered_map<Determinant, array<CItype, 2>> hash_det;

void cdfci::getDeterminantsVariational(
        Determinant& d, double epsilon, CItype ci1, CItype ci2,
        oneInt& int1, twoInt& int2, twoIntHeatBathSHM& I2hb,
        vector<int>& irreps, double coreE, double E0,
        hash_det& wavefunction, set<Determinant>& new_dets,
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
  int unpairedElecs = schd.enforceSeniority ?  d.numUnpairedElectrons() : 0;

  initiateRestrictions(schd, closed);
  //std::cout << "grow variational space" << std::endl;
  // mono-excited determinants
  //std::cout << "nopen, " << nopen << "nclosed, " << nclosed << std::endl;
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    //if (closed[i]/2 < schd.ncore || open[a]/2 >= schd.ncore+schd.nact) continue;
    //if (! satisfiesRestrictions(schd, closed[i], open[a])) continue;

    CItype integral = I2hb.Singles(open[a], closed[i]);//Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);
    //std::cout << "integral : " << integral << " epsilon : " << epsilon << std::endl;
    if (fabs(integral) > epsilon)
      if (closed[i]%2 == open[a]%2)
        integral = Hij_1Excite(open[a],closed[i],int1,int2, &closed[0], nclosed);

    //if (closed[i]%2 != open[a]%2) {
    //  integral = int1(open[a], closed[i])*schd.socmultiplier;
    //}
    //Have question about the SOC stuff

    // generate determinant if integral is above the criterion
    if (std::abs(integral) > epsilon ) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      //std::cout << "single excitation: " << di << std::endl;
      if(wavefunction.find(di) == wavefunction.end()) {
        std::array<double, 2>  val {0.0,0.0};
        wavefunction.insert({di, val});
        new_dets.insert(di);
      }
      //if (Determinant::Trev != 0) di.makeStandard();
    }
  } // ia

  // bi-excitated determinants
  if (std::abs(int2.maxEntry) < epsilon) return;
  // for all pairs of closed
  for (int ij=0; ij<nclosed*nclosed; ij++) {
    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    int X = max(I, J), Y = min(I, J);

    if (closed[i]/2 < schd.ncore || closed[j]/2 < schd.ncore) continue;

    int pairIndex = X*(X+1)/2+Y;
    size_t start = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex]   : I2hb.startingIndicesOppositeSpin[pairIndex];
    size_t end   = closed[i]%2==closed[j]%2 ? I2hb.startingIndicesSameSpin[pairIndex+1] : I2hb.startingIndicesOppositeSpin[pairIndex+1];
    float* integrals  = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinIntegrals : I2hb.oppositeSpinIntegrals;
    short* orbIndices = closed[i]%2==closed[j]%2 ?  I2hb.sameSpinPairs     : I2hb.oppositeSpinPairs;

    // for all HCI integrals
    //std::cout << "double excitation" << std::endl;
    for (size_t index=start; index<end; index++) {
      // if we are going below the criterion, break
      if (fabs(integrals[index]) < epsilon) break;

      // otherwise: generate the determinant corresponding to the current excitation
      int a = 2* orbIndices[2*index] + closed[i]%2, b= 2*orbIndices[2*index+1]+closed[j]%2;
      //double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);
      //if (abs(integrals[index]/(E0-Energyd)) <epsilon) continue;
      if (a/2 >= schd.ncore+schd.nact || b/2 >= schd.ncore+schd.nact) continue;
      if (! satisfiesRestrictions(schd, closed[i], closed[j], a, b)) continue;
      if (!(d.getocc(a) || d.getocc(b))) {
        Determinant di = d;
        di.setocc(a, true); di.setocc(b, true);di.setocc(closed[i],false); di.setocc(closed[j], false);
        if(wavefunction.find(di) == wavefunction.end()) {
          std::array<double, 2>  val {0.0,0.0};
          wavefunction.insert({di, val});
          new_dets.insert(di)
;        }
        //if (Determinant::Trev != 0) di.makeStandard();
      }
    } // heatbath integrals
  } // ij
  return;
} // end SHCIgetdeterminants::getDeterminantsVariational

vector<value_type> cdfci::getSubDets(value_type& d, hash_det& wfn, int nelec) {
  auto det = d.first;
  int norbs = det.norbs;
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  vector<int> closed(nelec, 0);
  vector<int> open(nopen, 0);
  det.getOpenClosed(open, closed);
    
  vector<value_type> result;
  result.push_back(d);
  //get all existing single excitationsgetDeterminantsVariational
  for (int ia=0; ia<nopen*nclosed; ia++) {
    int i=ia/nopen, a=ia%nopen;
    Determinant di = det;
    di.setocc(open[a], true);
    di.setocc(closed[i], false);
    if (wfn.find(di) != wfn.end()) {
      value_type val = std::make_pair(di, wfn[di]);
      result.push_back(val);
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
      if (wfn.find(di)!=wfn.end()){
        value_type val = std::make_pair(di, wfn[di]);
        result.push_back(val);
      }
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
    double abs_grad = fabs(x*norm+z);
    if (abs_grad >= max_abs_grad) {
      max_abs_grad = abs_grad;
      result = iter;
    }
  }
  return std::make_pair(result->first, result->second);
}

double cdfci::CoordinateUpdate(value_type& det_picked, hash_det & wfn, pair<double,double>& ene, vector<double> E0, oneInt& I1, twoInt& I2, double& coreE) {
  // coreE doesn't matter.
  double dx = 0.0;
  auto det = det_picked.first;
  auto x = det_picked.second[0];
  auto z = det_picked.second[1];
  // need to store a wavefunction norm here.
  double xx = ene.second;
  //std::cout << x << " z:" << z << " xx:" << xx << std::endl; 
  size_t orbDiff;

  auto dA = det.Energy(I1, I2, coreE);
  dA=-dA;
  // Line Search, cced from original cdfci code.
  auto p1 = xx - x * x - dA;
  auto q = z + dA * x;  //
  auto p3 = p1/3;
  auto q2 = q/2;
  auto d = p3 * p3 * p3 + q2 * q2;
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

void cdfci::civectorUpdate(vector<value_type>& column, hash_det& wfn, double dx, pair<double, double>& ene, oneInt& I1, twoInt& I2, double& coreE) {
  auto deti = column[0].first;
  //std::cout << "selected det: " << deti << std::endl;
  size_t orbDiff;
  //std::cout << ene.first << " " << ene.second << std::endl;
  for (auto entry : column) {
    auto detj = entry.first;
    double hij;
    auto iter = wfn.find(detj);
    if (detj == deti) {
      hij = deti.Energy(I1, I2, coreE);
      ene.first += hij * dx * dx+2*dx*hij*wfn[detj][0];
      ene.second += dx*dx + 2* wfn[detj][0]*dx;
      //std::cout << wfn[detj][1] << " " << wfn[detj][0] << " " << ene.first << std::endl;
      wfn[detj][0] += dx;
      wfn[detj][1] += hij*dx;
    }
    else {
      hij = Hij(deti, detj, I1, I2, coreE, orbDiff);
      double dz = dx*hij;
      ene.first += 2*iter->second[0]*dz;
      wfn[detj][1] += dz;
    }
    
    //std::cout << "det:" << detj << " civec:" << wfn[detj][0] << " hx:" << wfn[detj][1] << std::endl;
  }
  return; 
}

void cdfci::cdfciSolver(hash_det& wfn,set<Determinant>& new_dets, Determinant& hf, pair<double, double>& ene, oneInt& I1, twoInt& I2, double& coreE, vector<double>& E0, int nelec) {
    // first to initialize hx for new determinants
    auto startofCalc = getTime();
    for(auto i=new_dets.begin(); i!=new_dets.end(); i++) {
      Determinant di = *i;
      value_type thisDet = std::make_pair(di, wfn[di]);
      auto column = cdfci::getSubDets(thisDet, wfn, nelec);
      double z = 0.0;
      double x = 0.0;
      size_t orbDiff;
      for (auto entry : column) {
        auto detj = entry.first;
        double hij;
        if (detj == di) {
          z+=0.0;
        }
        else {
          hij = Hij(di, detj, I1, I2, coreE, orbDiff);
          auto xj = entry.second[0];
          z+= xj*hij;
        }
      }
      wfn[di]={0.0, z};
    }
    std::cout << "update determinants not included in previous variational iteration " << getTime()-startofCalc << std::endl;
    value_type thisDet =  std::make_pair(hf, wfn[hf]);
    auto sub = cdfci::getSubDets(thisDet, wfn, nelec);
    auto prev_ene=0.0;
    if (ene.second < 1e-100) prev_ene=0.0;
    else prev_ene = ene.first/ene.second;
    std::cout << std::endl;
    
    auto num_iter = wfn.size();
    for(int k=0; k<num_iter; k++) {
      for(int i=0; i<1000; i++) {
        auto dx = cdfci::CoordinateUpdate(thisDet, wfn, ene, E0, I1, I2, coreE);
        auto column = cdfci::getSubDets(thisDet, wfn, nelec);
        cdfci::civectorUpdate(column, wfn, dx, ene, I1, I2, coreE);
        
        //sub = cdfci::getSubDets(thisDet, wfn, nelec);
        thisDet = cdfci::CoordinatePickGcdGrad(column, ene.second);
      }
      auto curr_ene = ene.first/ene.second;
      std::cout<<"Current Variational Energy: " <<  std::setw(18) <<std::setprecision(10) << curr_ene << ". norm: " << ene.second << " prev variational energy: " << std::setw(18) << std::setprecision(10) << prev_ene << " time now " << std::setprecision(4) << getTime()-startofCalc << std::endl;
      //std::cout <<"ene:" << ene.first << " " << ene.second << std::endl; 
      if (std::abs(curr_ene-prev_ene) < 1e-10*1000.) {
        //hf = thisDet.first;
        break;
      }
      prev_ene = curr_ene;
    }
  return;
}

vector<double> cdfci::DoVariational(vector<MatrixXx>& ci, vector<Determinant> & Dets, schedule& schd, twoInt& I2, twoIntHeatBathSHM& I2HB, vector<int>& irrep, oneInt& I1, double& coreE, int nelec, bool DoRDM) {

  int proc = 0, nprocs = 1;

  if (schd.outputlevel > 0 && commrank == 0) {
    Time::print_time("start variation");
  }

  int nroots = ci.size();
  size_t norbs = I2.Direct.rows();
  int Norbs = norbs;
  
  // assume we keep one vector version of determinants and ci vectors, and one hash map version of it.

  pout << format("%4s %4s  %10s  %10.2e   %18s   %9s  %10s\n") % ("Iter") %
            ("Root") % ("Eps1 ") % ("#Var. Det.") % ("Energy") %
            ("#Davidson") % ("Time(s)");
  
  int prevSize = 0;
  int iterstart = 0;

  vector<Determinant> SortedDetsvec;
  SortedDetsvec = Dets;
  std::sort(SortedDetsvec.begin(), SortedDetsvec.end());
  int SortedDetsSize = SortedDetsvec.size();
  int DetsSize = Dets.size();

  hash_det wavefunction;
  for (int i=0; i<SortedDetsSize; i++) {
    double civec = ci[i](0);
    double norm = civec*civec;
    value_type value (SortedDetsvec[i], {0.0, 0.0});
    std::array<double, 2>  val {0.0,0.0};
    wavefunction.insert({SortedDetsvec[i], val});
  }
  //wavefunction[SortedDetsvec[0]]={1.0, SortedDetsvec[0].Energy(I1,I2,coreE)};
  //std::cout << wavefunction.begin()->first << std::endl;
  Determinant HF = Dets[0];
  CItype e0 = SortedDetsvec[0].Energy(I1,I2,coreE);
  vector<double> E0(nroots, abs(e0));
  //pair<double, double> ene = {e0, 1.0};
  //wavefunction[HF] = {1.0, e0};
  pair<double, double> ene={0.0,0.0};
  wavefunction[HF]={0.0,0.0};
  for (int iter = iterstart; iter < schd.epsilon1.size(); iter++) {
    double epsilon1 = schd.epsilon1[iter];
    
    CItype zero = 0.0;
    if (schd.outputlevel > 0) {
      pout << format("#-------------Iter=%4i---------------") % iter << endl;
    }

    // grow variational space
    auto SortedDetsSize = SortedDetsvec.size();
    set<Determinant> new_dets;
    std::cout << "time before grow variational space " << std::setw(10) << getTime()-startofCalc << std::endl;
    for (int i = 0; i < SortedDetsSize; i++) {
      auto norm = ene.second > 1e-10 ? ene.second : 1.0;
      auto civec = wavefunction[SortedDetsvec[i]][0];
      civec = civec / sqrt(norm);
      cdfci::getDeterminantsVariational(SortedDetsvec[i], epsilon1/abs(civec), civec, zero, I1, I2, I2HB, irrep, coreE, E0[0], wavefunction, new_dets, schd, 0, nelec);
    }
    std::cout << "time after grow variational space " 
    <<std:: setw(10) << getTime()-startofCalc << std::endl;
    auto&wfn = wavefunction;
    std::cout << "wfn size:  " << wfn.size() << std::endl;
    cdfci::cdfciSolver(wavefunction, new_dets, HF, ene, I1, I2, coreE, E0, nelec);
    //exit(0);
    
    //for(auto iter=wfn.begin(); iter!=wfn.end(); iter++) {
      //if (std::abs(iter->second[0]/ene.second) > 1e-3) 
      //std::cout << "det: " << iter->first << " civec: " << iter->second[0]/sqrt(ene.second)<< std::endl;
      //iter->second = {0.0,0.0};
    //}
    //ene={0.0,0.0};
    //exit(0);
    SortedDetsvec.clear(); 
    for (auto iter:wavefunction) {
      SortedDetsvec.push_back(iter.first);
    }
  }
  return E0;
}