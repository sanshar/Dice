#include "Determinants.h"
#include "CIPSIbasics.h"
#include "input.h"
#include "integral.h"
#include <vector>
#include "math.h"
#include "Hmult.h"
#include <tuple>
#include <map>
#include "Davidson.h"
#include "boost/format.hpp"
#include <fstream>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"

using namespace std;
using namespace Eigen;
using namespace boost;

void merge(Determinant *a, long low, long high, long mid, long* x, Determinant* c, long* cx)
{
  long i, j, k;
  i = low;
  k = low;
  j = mid + 1;
  while (i <= mid && j <= high)
    {
      if (a[i] < a[j])
        {
	  c[k] = a[i];
	  cx[k] = x[i];
	  k++;
	  i++;
        }
      else
        {
	  c[k] = a[j];
	  cx[k] = x[j];
	  k++;
	  j++;
        }
    }
  while (i <= mid)
    {
      c[k] = a[i];
      cx[k] = x[i];
      k++;
      i++;
    }
  while (j <= high)
    {
      c[k] = a[j];
      cx[k] = x[j];
      k++;
      j++;
    }
  for (i = low; i < k; i++)
    {
      a[i] =  c[i];
      x[i] = cx[i];
    }
}

void mergesort(Determinant *a, long low, long high, long* x, Determinant* c, long* cx)
{
  long mid;
  if (low < high)
    {
      mid=(low+high)/2;
      mergesort(a,low,mid, x, c, cx);
      mergesort(a,mid+1,high, x, c, cx);
      merge(a,low,high,mid, x, c, cx);
    }
  return;
}


int ipow(int base, int exp)
{
  int result = 1;
  while (exp)
    {
      if (exp & 1)
	result *= base;
      exp >>= 1;
      base *= base;
    }

  return result;
}


class StitchDEH {
 private:
  friend class boost::serialization::access;
  template<class Archive> 
  void serialize(Archive & ar, const unsigned int version) {
      ar & Det & Num & Energy;
  }

public:
  boost::shared_ptr<vector<Determinant> > Det;
  boost::shared_ptr<vector<double> > Num;
  boost::shared_ptr<vector<double> > Energy;
  
  StitchDEH() {
    Det = boost::shared_ptr<vector<Determinant> > (new vector<Determinant>() );
    Num = boost::shared_ptr<vector<double> > (new vector<double>() );
    Energy = boost::shared_ptr<vector<double> > (new vector<double>() );
  }

  StitchDEH(boost::shared_ptr<vector<Determinant> >pD, 
	    boost::shared_ptr<vector<double> >pNum, 
	    boost::shared_ptr<vector<double> >pE) 
  : Det(pD), Num(pNum), Energy(pE) {};

  void MergeSortAndRemoveDuplicates() {
    std::vector<Determinant> Detcopy = *Det;
    std::vector<double> Numcopy = *Num;
    std::vector<double> Ecopy = *Energy;

    long* detIndex =  new long[Detcopy.size()];
    long* detIndexcopy = new long[Detcopy.size()];
    for (size_t i=0; i<Detcopy.size(); i++)
      detIndex[i] = i;      
    mergesort(&Detcopy[0], 0, Detcopy.size()-1, detIndex, &( Det->operator[](0)), detIndexcopy);
    delete [] detIndexcopy;

    size_t uniqueSize = 0;
    Det->operator[](uniqueSize) = Detcopy[0];
    Num->operator[](uniqueSize) = Numcopy[ detIndex[0]];
    Energy->operator[](uniqueSize) = Ecopy[ detIndex[0]];
    for (size_t i=1; i <Detcopy.size(); i++) {
      if (!(Detcopy[i] == Detcopy[i-1])) {
	uniqueSize++;
	Det->operator[](uniqueSize) = Detcopy[i];
	Num->operator[](uniqueSize) = Numcopy[detIndex[i]];
	Energy->operator[](uniqueSize) = Ecopy[detIndex[i]];
      }
      else 
	Num->operator[](uniqueSize) += Numcopy[detIndex[i]];
    }
    Det->resize(uniqueSize+1);
    Num->resize(uniqueSize+1);
    Energy->resize(uniqueSize+1);
    
  }

  void RemoveDetsPresentIn(std::vector<Determinant>& SortedDets) {
      vector<Determinant>::iterator vec_it = SortedDets.begin();
      std::vector<Determinant> Detcopy = *Det;
      std::vector<double> Numcopy = *Num;
      std::vector<double> Ecopy = *Energy;

      size_t uniqueSize = 0;
      for (size_t i=0; i<Detcopy.size();) {
	if (Detcopy[i] < *vec_it) {
	  Det->operator[](uniqueSize) = Detcopy[i];
	  Num->operator[](uniqueSize) = Numcopy[i];
	  Energy->operator[](uniqueSize) = Ecopy[i];
	  i++; uniqueSize++;
	}
	else if (*vec_it < Detcopy[i] && vec_it != SortedDets.end())
	  vec_it ++;
	else if (*vec_it < Detcopy[i] && vec_it == SortedDets.end()) {
	  Det->operator[](uniqueSize) = Detcopy[i];
	  Num->operator[](uniqueSize) = Numcopy[i];
	  Energy->operator[](uniqueSize) = Ecopy[i];
	  i++; uniqueSize++;
	}
	else {
	  vec_it++; i++;
	}
      }
      Detcopy.clear(); Numcopy.clear(); Ecopy.clear();
      Det->resize(uniqueSize); Num->resize(uniqueSize); Energy->resize(uniqueSize);
  }

  void RemoveDuplicates() {
      std::vector<Determinant> Detcopy = *Det;
      std::vector<double> Numcopy = *Num;
      std::vector<double> Ecopy = *Energy;

      size_t uniqueSize = 0;
      Det->operator[](uniqueSize) = Detcopy[0];
      Num->operator[](uniqueSize) = Numcopy[ 0];
      Energy->operator[](uniqueSize) = Ecopy[ 0];
      for (size_t i=1; i <Detcopy.size(); i++) {
	if (!(Detcopy[i] == Detcopy[i-1])) {
	  uniqueSize++;
	  Det->operator[](uniqueSize) = Detcopy[i];
	  Num->operator[](uniqueSize) = Numcopy[i];
	  Energy->operator[](uniqueSize) = Ecopy[i];
	}
	else 
	  Num->operator[](uniqueSize) += Numcopy[i];
      }
      Det->resize(uniqueSize+1);
      Num->resize(uniqueSize+1);
      Energy->resize(uniqueSize+1);
  }

  void deepCopy(const StitchDEH& s) {
    *Det = *(s.Det);
    *Num = *(s.Num);
    *Energy = *(s.Energy);
  }
  
  void operator=(const StitchDEH& s) {
    Det = s.Det;
    Num = s.Num;
    Energy = s.Energy;
  }

  void clear() {
    Det->clear();
    Num->clear();
    Energy->clear();
  }

  void merge(const StitchDEH& s) {
    std::vector<Determinant> Detcopy = *Det;
    std::vector<double> Numcopy = *Num;
    std::vector<double> Ecopy = *Energy;
    
    Det->resize(Detcopy.size()+s.Det->size());
    Num->resize(Numcopy.size()+s.Det->size());
    Energy->resize(Ecopy.size()+s.Energy->size());

    size_t j = 0, k=0,l=0;
    while (j<Detcopy.size() && k <s.Det->size()) {
      if (Detcopy.operator[](j) < s.Det->operator[](k)) {
	Det->operator[](l) = Detcopy.operator[](j);
	Num->operator[](l) = Numcopy.operator[](j);
	Energy->operator[](l) = Ecopy.operator[](j);
	j++; l++;
      }
      else {
	Det->operator[](l) = s.Det->operator[](k);
	Num->operator[](l) = s.Num->operator[](k);
	Energy->operator[](l) = s.Energy->operator[](k);
	k++;l++;
      }
    }
    while (j<Detcopy.size()) {
      Det->operator[](l) = Detcopy.operator[](j);
      Num->operator[](l) = Numcopy.operator[](j);
      Energy->operator[](l) = Ecopy.operator[](j);
      j++; l++;
    }
    while (k<s.Det->size()) {
      Det->operator[](l) = s.Det->operator[](k);
      Num->operator[](l) = s.Num->operator[](k);
      Energy->operator[](l) = s.Energy->operator[](k);
      k++;l++;
    }
    
  }

};

class ElementWiseAddStitchDEH {
  public:
  StitchDEH operator()(const StitchDEH& s1, const StitchDEH& s2) {
    StitchDEH out;
    out.deepCopy(s1);
    out.merge(s2);  
    return out;
  }
};



//for each element in ci stochastic round to eps and put all the nonzero elements in newWts and their corresponding
//indices in Sample1
int CIPSIbasics::sample_round(MatrixXd& ci, double eps, std::vector<int>& Sample1, std::vector<double>& newWts){
  for (int i=0; i<ci.rows(); i++) {
    if (abs(ci(i,0)) > eps) {
      Sample1.push_back(i);
      newWts.push_back(ci(i,0));
    }
    else if (((double) rand() / (RAND_MAX))*eps < abs(ci(i,0))) {
      Sample1.push_back(i);
      newWts.push_back( eps*ci(i,0)/abs(ci(i,0)));
    }
  }
}


void CIPSIbasics::EvaluateAndStoreRDM(vector<vector<int> >& connections, vector<Determinant>& Dets, MatrixXd& ci,
				      vector<vector<size_t> >& orbDifference, int nelec, schedule& schd) {
  boost::mpi::communicator world;

  size_t norbs = Dets[0].norbs;

  //RDM(i,j,k,l) = a_i^\dag a_j^\dag a_l a_k
  //also i>=j and k>=l
  MatrixXd twoRDM(norbs*(norbs+1)/2, norbs*(norbs+1)/2);
  twoRDM *= 0.0;

  for (int i=0; i<Dets.size(); i++) {
    vector<int> closed(nelec, 0);
    vector<int> open(norbs-nelec,0);
    Dets[i].getOpenClosed(open, closed);

    //<Di| Gamma |Di>
    for (int n1=0; n1<nelec; n1++) {
      for (int n2=0; n2<n1; n2++) {
	int orb1 = closed[n1], orb2 = closed[n2];
	twoRDM(orb1*(orb1+1)/2 + orb2, orb1*(orb1+1)/2+orb2) += ci(i,0)*ci(i,0);
      }
    }


    for (int j=1; j<connections[i].size(); j++) {
      int d0=orbDifference[i][j]%norbs, c0=(orbDifference[i][j]/norbs)%norbs ;

      if (orbDifference[i][j]/norbs/norbs == 0) { //only single excitation
	for (int n1=0;n1<nelec; n1++) {
	  double sgn = 1.0;
	  int a=max(closed[n1],c0), b=min(closed[n1],c0), I=max(closed[n1],d0), J=min(closed[n1],d0); 
	  if (closed[n1] == d0) continue;
	  Dets[i].parity(min(d0,c0), max(d0,c0),sgn);
	  if (!( (closed[n1] > c0 && closed[n1] > d0) || (closed[n1] < c0 && closed[n1] < d0))) sgn *=-1.;
	  twoRDM(a*(a+1)/2+b, I*(I+1)/2+J) += sgn*ci(connections[i][j],0)*ci(i,0);
	  twoRDM(I*(I+1)/2+J, a*(a+1)/2+b) += sgn*ci(connections[i][j],0)*ci(i,0);
	}
      }
      else {
	int d1=(orbDifference[i][j]/norbs/norbs)%norbs, c1=(orbDifference[i][j]/norbs/norbs/norbs)%norbs ;
	double sgn = 1.0;

	Dets[i].parity(d1,d0,c1,c0,sgn);

	twoRDM(c1*(c1+1)/2+c0, d1*(d1+1)/2+d0) += sgn*ci(connections[i][j],0)*ci(i,0);
	twoRDM(d1*(d1+1)/2+d0, c1*(c1+1)/2+c0) += sgn*ci(connections[i][j],0)*ci(i,0);
      }
    }
  }

  int nSpatOrbs = norbs/2;
  MatrixXd s2RDM(nSpatOrbs*nSpatOrbs, nSpatOrbs*nSpatOrbs);
  s2RDM *= 0.0;


  for (int n1=0; n1<nSpatOrbs; n1++)
  for (int n2=0; n2<nSpatOrbs; n2++)
  for (int n3=0; n3<nSpatOrbs; n3++)
  for (int n4=0; n4<nSpatOrbs; n4++)
  {
    double sgn = 1.0;
    int N1 = 2*max(n1,n2), N2=2*min(n1,n2), N3=2*max(n3,n4), N4=2*min(n3,n4);
    if(( (n1>=n2 && n3<n4) || (n1<n2 && n3>=n4))) sgn = -1.0; 
    s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4) += sgn*twoRDM( N1*(N1+1)/2+N2, N3*(N3+1)/2+N4);

    sgn = 1.0;
    N1 = max(2*n1+1,2*n2); N2=min(2*n1+1,2*n2); N3=max(2*n3+1,2*n4); N4=min(2*n3+1,2*n4);
    if(!( (2*n1+1>2*n2 && 2*n3+1>2*n4) || (2*n1+1<2*n2 && 2*n3+1<2*n4))) sgn = -1.0; 
    s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4) += sgn*twoRDM( N1*(N1+1)/2+N2, N3*(N3+1)/2+N4);

    sgn = 1.0;
    N1 = max(2*n1,2*n2+1); N2=min(2*n1,2*n2+1); N3=max(2*n3,2*n4+1); N4=min(2*n3,2*n4+1);
    if(!( (2*n1>=2*n2+1 && 2*n3>=2*n4+1) || (2*n1<2*n2+1 && 2*n3<2*n4+1))) sgn = -1.0; 
    s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4) += sgn*twoRDM( N1*(N1+1)/2+N2, N3*(N3+1)/2+N4);

    sgn = 1.0;
    N1 = 2*max(n1,n2)+1; N2=2*min(n1,n2)+1; N3=2*max(n3,n4)+1; N4=2*min(n3,n4)+1;
    if(( (n1>=n2 && n3<n4) || (n1<n2 && n3>=n4))) sgn = -1.0; 
    s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4) += sgn*twoRDM( N1*(N1+1)/2+N2, N3*(N3+1)/2+N4);
  }

  {
    char file [5000];
    sprintf (file, "%s/%d-spinRDM.bkp" , schd.prefix.c_str(), world.rank() );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << twoRDM;
  }
  {
    char file [5000];
    sprintf (file, "%s/%d-spatialRDM.bkp" , schd.prefix.c_str(), world.rank() );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << s2RDM;
  }

}

void CIPSIbasics::setUpAliasMethod(MatrixXd& ci, double& cumulative, std::vector<int>& alias, std::vector<double>& prob) {
  alias.resize(ci.rows());
  prob.resize(ci.rows());
  
  std::vector<double> larger, smaller;
  for (int i=0; i<ci.rows(); i++) {
    prob[i] = abs(ci(i,0))*ci.rows()/cumulative;
    if (prob[i] < 1.0)
      smaller.push_back(i);
    else
      larger.push_back(i);
  }

  while (larger.size() >0 && smaller.size() >0) {
    int l = larger[larger.size()-1]; larger.pop_back();
    int s = smaller[smaller.size()-1]; smaller.pop_back();
    
    alias[s] = l;
    prob[l] = prob[l] - (1.0 - prob[s]);
    if (prob[l] < 1.0)
      smaller.push_back(l);
    else
      larger.push_back(l);
  }
}

int CIPSIbasics::sample_N2_alias(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts, std::vector<int>& alias, std::vector<double>& prob) {

  int niter = Sample1.size(); //Sample1.resize(0); newWts.resize(0);

  int sampleIndex = 0;
  for (int index = 0; index<niter; index++) {
    int detIndex = floor(1.* ((double) rand() / (RAND_MAX))*ci.rows() );

    double rand_no = ((double) rand()/ (RAND_MAX));
    if (rand_no >= prob[detIndex]) 
      detIndex = alias[detIndex];

    std::vector<int>::iterator it = find(Sample1.begin(), Sample1.end(), detIndex);
    if (it == Sample1.end()) {
      Sample1[sampleIndex] = detIndex;
      newWts[sampleIndex] = ci(detIndex,0) < 0. ? -cumulative : cumulative;
      sampleIndex++;
    }
    else {
      newWts[distance(Sample1.begin(), it) ] += ci(detIndex,0) < 0. ? -cumulative : cumulative;
    }
  }

  for (int i=0; i<niter; i++)
    newWts[i] /= niter;
  return sampleIndex;
}

int CIPSIbasics::sample_N2(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts){
  double prob = 1.0;
  int niter = Sample1.size();
  int totalSample = 0;
  for (int index = 0; index<niter; ) {

    double rand_no = ((double) rand() / (RAND_MAX))*cumulative;
    for (int i=0; i<ci.rows(); i++) {
      if (rand_no < abs(ci(i,0))) {
	std::vector<int>::iterator it = find(Sample1.begin(), Sample1.end(), i);
	if (it == Sample1.end()) {
	  Sample1[index] = i;
	  newWts[index] = ci(i,0) < 0. ? -cumulative : cumulative;
	  index++; totalSample++;
	}
	else {
	  //if (Sample1[distance(Sample1.begin(), it) ] != i) {cout << i<<" "<<*it <<endl; exit(0);}
	  //int oldindex = distance(Sample1.begin(), it);
	  //cout << oldindex<<"  "<<newWts[oldindex]<<"  "<<cumulative/Sample1.size()<<"  "<<Sample1.size()<<endl;
	  newWts[ distance(Sample1.begin(), it) ] += ci(i,0) < 0. ? -cumulative : cumulative;
	  totalSample++;
	}
	break;
      }
      rand_no -= abs(ci(i,0));
    }
  }

  for (int i=0; i<niter; i++)
    newWts[i] /= totalSample;
  return totalSample;
}

int CIPSIbasics::sample_N(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts){
  double prob = 1.0;
  int niter = Sample1.size();

  for (int index = 0; index<niter; ) {

    double rand_no = ((double) rand() / (RAND_MAX))*cumulative;
    for (int i=0; i<ci.rows(); i++) {
      if (rand_no < abs(ci(i,0))) {

	Sample1[index] = i;
	newWts[index] = ci(i,0) < 0. ? -cumulative/Sample1.size() : cumulative/Sample1.size();
	index++;
	break;
      }
      rand_no -= abs(ci(i,0));
    }
  }

}


void CIPSIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
								    twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  std::ofstream ofs(file);

  double epsilon2 = schd.epsilon2;
  schd.epsilon2 = schd.epsilon2Large;
  double EptLarge = DoPerturbativeDeterministic(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);

  schd.epsilon2 = epsilon2;

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  int niter = 1000000;
  //double eps = 0.001;
  int Nsample = schd.SampleN;
  double AvgenergyEN = 0.0;
  double AverageDen = 0.0;
  int currentIter = 0;
  int sampleSize = 0;
  int num_thrds = omp_get_max_threads();
  
  double cumulative = 0.0;
  for (int i=0; i<ci.rows(); i++)
    cumulative += abs(ci(i,0));

  std::vector<int> alias; std::vector<double> prob;
  setUpAliasMethod(ci, cumulative, alias, prob);
#pragma omp parallel for schedule(dynamic) 
  for (int iter=0; iter<niter; iter++) {
      //cout << norbs<<"  "<<nelec<<endl;
    char psiArray[norbs]; 
    vector<int> psiClosed(nelec,0); 
    vector<int> psiOpen(norbs-nelec,0);
    //char psiArray[norbs];
    std::vector<double> wts1(Nsample,0.0); std::vector<int> Sample1(Nsample,-1);
    
    //int Nmc = sample_N2(ci, cumulative, Sample1, wts1);
    int distinctSample = sample_N2_alias(ci, cumulative, Sample1, wts1, alias, prob);
    int Nmc = Nsample;
    double norm = 0.0;
    
    //map<Determinant, std::tuple<double,double,double, double, double> > Psi1ab; 
    std::vector<Determinant> Psi1; std::vector<double>  numerator1A, numerator2A;
    std::vector<double>  numerator1B, numerator2B;
    std::vector<double>  det_energy;
    for (int i=0; i<distinctSample; i++) {
      int I = Sample1[i];
      CIPSIbasics::getDeterminants2Epsilon(Dets[I], abs(schd.epsilon2/ci(I,0)), abs(schd.epsilon2Large/ci(I,0)), wts1[i], ci(I,0), I1, I2, I2HB, irrep, coreE, E0, Psi1, numerator1A, numerator2A, numerator1B, numerator2B, det_energy, schd, Nmc, nelec);
    }

      std::vector<long> index(Psi1.size());
      for (size_t k=0; k<Psi1.size(); k++)
	index[k] = k;
      long* indexcopy = new long[Psi1.size()];
      Determinant* dc = new Determinant[Psi1.size()];
      mergesort(&Psi1[0], 0, Psi1.size()-1, &index[0], dc, indexcopy);
      delete [] indexcopy;
      delete [] dc;

      double currentNum1A=0., currentNum2A=0.;
      double currentNum1B=0., currentNum2B=0.;
      vector<Determinant>::iterator vec_it = SortedDets.begin();
      double energyEN = 0.0, energyENLargeEps = 0.0;

      for (int i=0;i<Psi1.size();) {
	if (Psi1[i] < *vec_it) {
	  currentNum1A += numerator1A[index[i]];
	  currentNum2A += numerator2A[index[i]];
	  currentNum1B += numerator1B[index[i]];
	  currentNum2B += numerator2B[index[i]];
	  if ( i == Psi1.size()-1) {
	    energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) - currentNum2A)/(det_energy[index[i]] - E0);
	    energyENLargeEps += (currentNum1B*currentNum1B*Nmc/(Nmc-1) - currentNum2B)/(det_energy[index[i]] - E0);
	  }
	  else if (!(Psi1[i] == Psi1[i+1])) {
	    energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) - currentNum2A)/(det_energy[index[i]] - E0);
	    energyENLargeEps += (currentNum1B*currentNum1B*Nmc/(Nmc-1) - currentNum2B)/(det_energy[index[i]] - E0);
	    currentNum1A = 0.;
	    currentNum2A = 0.;
	    currentNum1B = 0.;
	    currentNum2B = 0.;
	  }
	  i++;
	}
	else if (*vec_it <Psi1[i] && vec_it != SortedDets.end())
	  vec_it++;
	else if (*vec_it <Psi1[i] && vec_it == SortedDets.end()) {
	  currentNum1A += numerator1A[index[i]];
	  currentNum2A += numerator2A[index[i]];
	  currentNum1B += numerator1B[index[i]];
	  currentNum2B += numerator2B[index[i]];
	  if ( i == Psi1.size()-1) {
	    energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) - currentNum2A)/(det_energy[index[i]] - E0);
	    energyENLargeEps += (currentNum1B*currentNum1B*Nmc/(Nmc-1) - currentNum2B)/(det_energy[index[i]] - E0);
	  }
	  else if (!(Psi1[i] == Psi1[i+1])) {
	    energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) - currentNum2A)/(det_energy[index[i]] - E0);
	    energyENLargeEps += (currentNum1B*currentNum1B*Nmc/(Nmc-1) - currentNum2B)/(det_energy[index[i]] - E0);
	    currentNum1A = 0.;
	    currentNum2A = 0.;
	    currentNum1B = 0.;
	    currentNum2B = 0.;
	  }
	  i++;
	}
	else {
	  if (Psi1[i] == Psi1[i+1])
	    i++;
	  else {
	    vec_it++; i++;
	  }
	}
      }

      sampleSize = distinctSample;
    
    
#pragma omp critical 
    {
      if (mpigetrank() == 0) {
	AvgenergyEN += -energyEN+energyENLargeEps+EptLarge; currentIter++;
	std::cout << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	  %(currentIter) % (E0-energyEN+energyENLargeEps+EptLarge) % (norm) % (E0+AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	cout << endl;
      }
      else {
	AvgenergyEN += -energyEN+energyENLargeEps+EptLarge; currentIter++;
	ofs << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	  %(currentIter) % (E0-energyEN+energyENLargeEps+EptLarge) % (norm) % (E0+AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	ofs << endl;
	
      }
    }
  }
  ofs.close();
  
}

void CIPSIbasics::DoPerturbativeStochastic2SingleList(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
						      twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  std::ofstream ofs(file);

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 1000000;
    //double eps = 0.001;
    int Nsample = schd.SampleN;
    double AvgenergyEN = 0.0;
    double AverageDen = 0.0;
    int currentIter = 0;
    int sampleSize = 0;
    int num_thrds = omp_get_max_threads();

    double cumulative = 0.0;
    for (int i=0; i<ci.rows(); i++)
      cumulative += abs(ci(i,0));

    std::vector<int> alias; std::vector<double> prob;
    setUpAliasMethod(ci, cumulative, alias, prob);
#pragma omp parallel for schedule(dynamic) 
    for (int iter=0; iter<niter; iter++) {
      //cout << norbs<<"  "<<nelec<<endl;
      char psiArray[norbs]; 
      vector<int> psiClosed(nelec,0); 
      vector<int> psiOpen(norbs-nelec,0);
      //char psiArray[norbs];
      std::vector<double> wts1(Nsample,0.0); std::vector<int> Sample1(Nsample,-1);

      //int Nmc = sample_N2(ci, cumulative, Sample1, wts1);
      int distinctSample = sample_N2_alias(ci, cumulative, Sample1, wts1, alias, prob);
      int Nmc = Nsample;
      double norm = 0.0;
      
      map<Determinant, std::tuple<double,double,double> > Psi1ab; 
      size_t initSize = 100000;
      std::vector<Determinant> Psi1; std::vector<double>  numerator1, numerator2;
      std::vector<double>  det_energy;
      Psi1.reserve(initSize); numerator1.reserve(initSize); numerator2.reserve(initSize); det_energy.reserve(initSize);
      for (int i=0; i<distinctSample; i++) {
       int I = Sample1[i];
       CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], ci(I,0), I1, I2, I2HB, irrep, coreE, E0, Psi1, numerator1, numerator2, det_energy, schd, Nmc, nelec);
      }


      std::vector<long> index(Psi1.size());
      for (size_t k=0; k<Psi1.size(); k++)
	index[k] = k;
      long* indexcopy = new long[Psi1.size()];
      Determinant* dc = new Determinant[Psi1.size()];
      mergesort(&Psi1[0], 0, Psi1.size()-1, &index[0], dc, indexcopy);
      delete [] indexcopy;
      delete [] dc;

      double currentNum1=0., currentNum2=0.;
      vector<Determinant>::iterator vec_it = SortedDets.begin();
      double energyEN = 0.0;

      for (int i=0;i<Psi1.size();) {
	if (Psi1[i] < *vec_it) {
	  currentNum1 += numerator1[index[i]];
	  currentNum2 += numerator2[index[i]];
	  if ( i == Psi1.size()-1) 
	    energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) - currentNum2)/(det_energy[index[i]] - E0);
	  else if (!(Psi1[i] == Psi1[i+1])) {
	    energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) - currentNum2)/(det_energy[index[i]] - E0);
	    currentNum1 = 0.;
	    currentNum2 = 0.;
	  }
	  i++;
	}
	else if (*vec_it <Psi1[i] && vec_it != SortedDets.end())
	  vec_it++;
	else if (*vec_it <Psi1[i] && vec_it == SortedDets.end()) {
	  currentNum1 += numerator1[index[i]];
	  currentNum2 += numerator2[index[i]];
	  if ( i == Psi1.size()-1) 
	    energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) - currentNum2)/(det_energy[index[i]] - E0);
	  else if (!(Psi1[i] == Psi1[i+1])) {
	    energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) - currentNum2)/(det_energy[index[i]] - E0);
	    currentNum1 = 0.;
	    currentNum2 = 0.;
	  }
	  i++;
	}
	else {
	  if (Psi1[i] == Psi1[i+1])
	    i++;
	  else {
	    vec_it++; i++;
	  }
	}
      }

      sampleSize = distinctSample;

#pragma omp critical 
      {
	if (mpigetrank() == 0) {
	  AvgenergyEN += energyEN; currentIter++;
	  std::cout << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  cout << endl;
	}

	else {
	  AvgenergyEN += energyEN; currentIter++;
	  ofs << format("%6i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  ofs << endl;

	}
      }
    }
    ofs.close();

}


double CIPSIbasics::DoPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					      twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

  boost::mpi::communicator world;
  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  char psiArray[norbs]; vector<int> psiClosed(nelec,0), psiOpen(norbs-nelec,0);
  //char psiArray[norbs]; int psiOpen[nelec], psiClosed[norbs-nelec];
  double energyEN = 0.0;
  int num_thrds = omp_get_max_threads();
  
  
  std::vector<StitchDEH> uniqueDEH(num_thrds);
#pragma omp parallel 
  {
    for (int i=0; i<Dets.size(); i++) {
      if (i%(omp_get_num_threads()*mpigetsize()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) {continue;}
      CIPSIbasics::getDeterminants(Dets[i], abs(schd.epsilon2/ci(i,0)), ci(i,0), 0.0, 
				   I1, I2, I2HB, irrep, coreE, E0, 
				   *uniqueDEH[omp_get_thread_num()].Det, 
				   *uniqueDEH[omp_get_thread_num()].Num, 
				   *uniqueDEH[omp_get_thread_num()].Energy, 
				   schd,0, nelec);
      if (i%100000 == 0 && omp_get_thread_num()==0 && mpigetrank() == 0) cout <<"# "<<i<<endl;
    }
    
    
    if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#Before sort "<<getTime()-startofCalc<<endl;
    
    uniqueDEH[omp_get_thread_num()].MergeSortAndRemoveDuplicates();
    uniqueDEH[omp_get_thread_num()].RemoveDetsPresentIn(SortedDets);
    
    if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#Unique determinants "<<getTime()-startofCalc<<"  "<<endl;
    
    
    //Now merge threads to make a list of unique determinants. Note we have still not merged the processors
    //The code is messy because we are using a method that is Log N in the number of threads
    for (int level=0; level<ceil(log2(omp_get_num_threads())); level++) {
#pragma omp barrier
      if (omp_get_thread_num()%ipow(2,level+1) == 0 && omp_get_thread_num() + ipow(2,level) < omp_get_num_threads() ) {	
	int other_thrd = omp_get_thread_num()+ipow(2,level);
	uniqueDEH[omp_get_thread_num()].merge(uniqueDEH[other_thrd]);
	uniqueDEH[other_thrd].clear();
	uniqueDEH[omp_get_thread_num()].RemoveDuplicates();
      }	        
    }
  }
  uniqueDEH.resize(1);
  
  if (mpigetrank() == 0 ) cout << "#Before mpi split "<<getTime()-startofCalc<<"  "<<uniqueDEH[0].Det->size()<<endl;
  if (mpigetsize() > 1) {
    vector<Determinant>& uniqueDets = *uniqueDEH[0].Det;
    vector<double>& uniqueNumerator = *uniqueDEH[0].Num;
    vector<double>& uniqueEnergy = *uniqueDEH[0].Energy;
    
    vector<StitchDEH> DataPerProc(mpigetsize());
    for (int i=0; i<uniqueDets.size(); i++) {
      //if (mpigetrank() == 0) cout <<i<<"  "<< uniqueDets[i]<<endl;
      int proc = uniqueDets[i].getLexicalOrder()%mpigetsize() ;
      //int proc = uniqueDets[i].repr[0]%mpigetsize() ;
      DataPerProc[proc].Det->push_back(uniqueDets[i]);
      DataPerProc[proc].Num->push_back(uniqueNumerator[i]);
      DataPerProc[proc].Energy->push_back(uniqueEnergy[i]);
    }
    uniqueDEH[0].clear();
    
    if (mpigetrank() == 0) cout << "#Before mpi reduce "<<DataPerProc[mpigetrank()].Det->size()<<"  "<<getTime()-startofCalc<<"  "<<endl;
    for (int proc = 0; proc<mpigetsize(); proc++) {
      mpi::reduce(world, DataPerProc[proc], uniqueDEH[0], ElementWiseAddStitchDEH(), proc);
      DataPerProc[proc].clear();
    }
    if (mpigetrank() == 0) cout << "#after mpi reduce, before merge "<<DataPerProc[mpigetrank()].Det->size()<<"  "<<getTime()-startofCalc<<"  "<<endl;
    uniqueDEH[0].RemoveDuplicates();
  }
  
  vector<Determinant>& uniqueDets = *uniqueDEH[0].Det;
  vector<double>& uniqueNumerator = *uniqueDEH[0].Num;
  vector<double>& uniqueEnergy = *uniqueDEH[0].Energy;
  double totalPT=0.0;

  size_t numDets=0, numLocalDets=uniqueDets.size();
  mpi::all_reduce(world, numLocalDets, numDets, std::plus<size_t>());
  if (mpigetrank() == 0) cout <<"#num dets "<<numDets<<endl;
#pragma omp parallel
  {
    double PTEnergy = 0.0;
    for (size_t i=0; i<uniqueDets.size();i++) {
      if (i%(omp_get_num_threads()) != omp_get_thread_num()) continue;
      PTEnergy += uniqueNumerator[i]*uniqueNumerator[i]/(E0-uniqueEnergy[i]);
    }
#pragma omp critical
    {
      totalPT += PTEnergy;
    }
    
  }
  uniqueDEH[0].clear();
  double finalE = 0.;
  mpi::all_reduce(world, totalPT, finalE, std::plus<double>());
  
  if (mpigetrank() == 0) cout << "#Done energy "<<finalE<<"  "<<getTime()-startofCalc<<endl;
  return finalE;
}

void MakeHfromHelpers(std::map<HalfDet, std::vector<int> >& BetaN,
		      std::map<HalfDet, std::vector<int> >& AlphaNm1,
		      std::vector<Determinant>& Dets,
		      int StartIndex,
		      std::vector<std::vector<int> >&connections,
		      std::vector<std::vector<double> >& Helements,
		      char* detChar,
		      int Norbs,
		      oneInt& I1,
		      twoInt& I2,
		      double& coreE,
		      std::vector<std::vector<size_t> >& orbDifference,
		      bool DoRDM=false) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int nprocs= mpigetsize(), proc = mpigetrank();

  size_t norbs = Norbs;

#pragma omp parallel 
  {
    for (size_t k=StartIndex; k<Dets.size(); k++) {
      if (k%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;
      connections[k].push_back(k);
      double hij = Energy(&detChar[norbs*k], Norbs, I1, I2, coreE);
      Helements[k].push_back(hij);
      if (DoRDM) orbDifference[k].push_back(0);
    }
  }

  std::map<HalfDet, std::vector<int> >::iterator ita = BetaN.begin();
  int index = 0;
  for (; ita!=BetaN.end(); ita++) {
    std::vector<int>& detIndex = ita->second;
    int localStart = detIndex.size();
    for (int j=0; j<detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }
    
#pragma omp parallel 
    {
      for (int k=localStart; k<detIndex.size(); k++) {
      
	if (detIndex[k]%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;
	//if (detIndex[j]%omp_get_num_threads() != omp_get_thread_num()) continue;
	
	for(int j=0; j<k; j++) {
	  size_t J = detIndex[j];size_t K = detIndex[k];
	  if (Dets[J].connected(Dets[K])) 	  {
	    connections[K].push_back(J);
	    //double hij = Hij(&detChar[norbs*J], &detChar[norbs*K], Norbs, I1, I2, coreE);
	    size_t orbDiff;
	    double hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
	    Helements[K].push_back(hij);
	    if (DoRDM) 
	      orbDifference[K].push_back(orbDiff);
	  }
	}
      }
    }
    index++;
    if (index%100000 == 0 && index!= 0) {pout <<"#bn "<<index<<endl;}
  }

  pout <<"# "<< Dets.size()<<"  "<<BetaN.size()<<"  "<<AlphaNm1.size()<<endl;
  ita = AlphaNm1.begin();
  index = 0;
  for (; ita!=AlphaNm1.end(); ita++) {
    std::vector<int>& detIndex = ita->second;
    int localStart = detIndex.size();
    for (int j=0; j<detIndex.size(); j++)
      if (detIndex[j] >= StartIndex) {
	localStart = j; break;
      }

#pragma omp parallel 
    {
      for (int k=localStart; k<detIndex.size(); k++) {
	if (detIndex[k]%(nprocs*omp_get_num_threads()) != proc*omp_get_num_threads()+omp_get_thread_num()) continue;

	for(int j=0; j<k; j++) {
	  size_t J = detIndex[j];size_t K = detIndex[k];
	  if (Dets[J].connected(Dets[K]) ) {
	    if (find(connections[K].begin(), connections[K].end(), J) == connections[K].end()){
	      connections[K].push_back(J);
	      size_t orbDiff;
	      double hij = Hij(Dets[J], Dets[K], I1, I2, coreE, orbDiff);
	      //double hij = Hij(&detChar[norbs*J], &detChar[norbs*K], Norbs, I1, I2, coreE);
	      Helements[K].push_back(hij);
	      if (DoRDM) 
		orbDifference[K].push_back(orbDiff);
	    }
	  }
	}
      }
    }
    index++;
    if (index%100000 == 0 && index!= 0) {pout <<"#an-1 "<<index<<endl;}
  }
  pout << format("#AlphaN-1 %49.2f\n")
      % (getTime()-startofCalc);


    
}
  
void PopulateHelperLists(std::map<HalfDet, std::vector<int> >& BetaN,
			 std::map<HalfDet, std::vector<int> >& AlphaNm1,
			 std::vector<Determinant>& Dets,
			 int StartIndex) {
  pout << format("#Making Helpers %43.2f\n")
      % (getTime()-startofCalc);
  for (int i=StartIndex; i<Dets.size(); i++) {
    HalfDet da = Dets[i].getAlpha(), db = Dets[i].getBeta();

    BetaN[db].push_back(i);

    int norbs = 64*DetLen;
    std::vector<int> closeda(norbs);//, closedb(norbs);
    int ncloseda = da.getClosed(closeda);
    //int nclosedb = db.getClosed(closedb);

    
    for (int j=0; j<ncloseda; j++) {
      da.setocc(closeda[j], false);
      AlphaNm1[da].push_back(i);
      da.setocc(closeda[j], true);
    }
  }

}

//this takes in a ci vector for determinants placed in Dets
//it then does a CIPSI varitional calculation and the resulting
//ci and dets are returned here
//at input usually the Dets will just have a HF or some such determinant
//and ci will be just 1.0
double CIPSIbasics::DoVariational(MatrixXd& ci, vector<Determinant>& Dets, schedule& schd,
				  twoInt& I2, twoIntHeatBath& I2HB, vector<int>& irrep, oneInt& I1, double& coreE
				  , int nelec, bool DoRDM) {

  std::map<HalfDet, std::vector<int> > BetaN, AlphaNm1;
  PopulateHelperLists(BetaN, AlphaNm1, Dets, 0);


#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int num_thrds = omp_get_max_threads();

  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());

  size_t norbs = 2.*I2.Direct.rows();
  int Norbs = norbs;
  std::vector<char> detChar(norbs); Dets[0].getRepArray(&detChar[0]);
  double E0 = Energy(&detChar[0], Norbs, I1, I2, coreE);

  pout << "#HF = "<<E0<<std::endl;

  //this is essentially the hamiltonian, we have stored it in a sparse format
  std::vector<std::vector<int> > connections(1, std::vector<int>(1,0));
  std::vector<std::vector<double> > Helements(1, std::vector<double>(1,E0));
  std::vector<std::vector<size_t> > orbDifference(1, std::vector<size_t>(1, 0));
  if (mpigetrank() != 0) {connections.resize(0); Helements.resize(0);orbDifference.resize(0);}


  //keep the diagonal energies of determinants so we only have to generated
  //this for the new determinants in each iteration and not all determinants
  MatrixXd diagOld(1,1); diagOld(0,0) = E0;
  int prevSize = 0;

  int iterstart = 0;

  if (schd.restart || schd.fullrestart) {
    bool converged;
    readVariationalResult(iterstart, ci, Dets, SortedDets, diagOld, connections, Helements, E0, converged, schd, BetaN, AlphaNm1);
    pout << format("# %4i  %10.2e  %10.2e   %14.8f  %10.2f\n") 
      %(iterstart) % schd.epsilon1[iterstart] % Dets.size() % E0 % (getTime()-startofCalc);
    iterstart++;
    if (schd.onlyperturbative)
      return E0;

    detChar.resize(Dets.size()*norbs);
    for (int i=0; i<Dets.size(); i++) 
      Dets[i].getRepArray(&detChar[i*norbs]);
    if (converged && !schd.fullrestart) {
      pout << "# restarting from a converged calculation, moving to perturbative part.!!"<<endl;
      return E0;
    }
  }


  //do the variational bit
  for (int iter=iterstart; iter<schd.epsilon1.size(); iter++) {
    double epsilon1 = schd.epsilon1[iter];
    std::vector<vector<Determinant> > newDets(num_thrds); //also include the connection magnitude so we can calculate the pt

#pragma omp parallel 
    {
      for (int i=0; i<SortedDets.size(); i++) {
	if (i%(mpigetsize()*omp_get_num_threads()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) continue;
	getDeterminants(Dets[i], abs(epsilon1/ci(i,0)), ci(i,0), 0.0, I1, I2, I2HB, irrep, coreE, E0, newDets[omp_get_thread_num()], schd, 0, nelec);
      }

      long* indices = new long[newDets[omp_get_thread_num()].size()];
      long* ic = new long[newDets[omp_get_thread_num()].size()];
      Determinant* dc = new Determinant[newDets[omp_get_thread_num()].size()];
      mergesort(&newDets[omp_get_thread_num()][0], 0, newDets[omp_get_thread_num()].size()-1, indices, dc, ic);
      delete [] indices;delete [] ic;delete [] dc;

      //std::sort(newDets[omp_get_thread_num()].begin(), newDets[omp_get_thread_num()].end());
      newDets[omp_get_thread_num()].erase(unique(newDets[omp_get_thread_num()].begin(), newDets[omp_get_thread_num()].end()),  newDets[omp_get_thread_num()].end());

      for (int level=0; level<ceil(log2(omp_get_num_threads())); level++) {
#pragma omp barrier
	if (omp_get_thread_num()%ipow(2,level+1) == 0 && omp_get_thread_num() + ipow(2,level) < omp_get_num_threads() ) {

	  int other_thrd = omp_get_thread_num()+ipow(2,level);
	  int this_thrd = omp_get_thread_num();
	  vector<Determinant> merged;
	  std::merge(newDets[this_thrd].begin(), newDets[this_thrd].end(), newDets[other_thrd].begin(), newDets[other_thrd].end(), std::insert_iterator<vector<Determinant> >(merged, merged.end()));

	  merged.erase(std::unique(merged.begin(), merged.end()), merged.end() );
	  newDets[this_thrd] = merged;
	  newDets[other_thrd].clear();
	}	  
      }
    }
    newDets.resize(1);


#ifndef SERIAL
    for (int level = 0; level <ceil(log2(mpigetsize())); level++) {

      if (mpigetrank()%ipow(2, level+1) == 0 && mpigetrank() + ipow(2, level) < mpigetsize()) {
	vector<Determinant> newDetsRecv;
	size_t newdetsrecvsize=-1;
	int getproc = mpigetrank()+ipow(2,level);
	world.recv(getproc, getproc, newdetsrecvsize);
	if (newdetsrecvsize > 0) {
	  world.recv(getproc, getproc, newDetsRecv);
	  vector<Determinant> merged;
	  std::merge(newDets[0].begin(), newDets[0].end(), newDetsRecv.begin(), newDetsRecv.end(), 
		     std::insert_iterator<vector<Determinant> >(merged, merged.end())
		     );

	  merged.erase(std::unique(merged.begin(), merged.end()), 
		     merged.end()
		     );
	  newDets[0] = merged;
	}
      }
      else if ( mpigetrank()%ipow(2, level+1) == 0 && mpigetrank() + ipow(2, level) > mpigetsize()) {
	continue ;
      } 
      else if ( mpigetrank()%ipow(2, level) == 0) {
	int toproc = mpigetrank()-ipow(2,level);
	world.send(toproc, mpigetrank(), newDets[0].size());
	if (newDets[0].size() > 0) 
	  world.send(toproc, mpigetrank(), newDets[0]);
	newDets[0].resize(0);
      }
    }


    mpi::broadcast(world, newDets[0], 0);
#endif

    vector<Determinant>::iterator vec_it = SortedDets.begin();
    for (vector<Determinant>::iterator it=newDets[0].begin(); it!=newDets[0].end(); ) {
      if (schd.excitation != 1000 ) {
	if (it->ExcitationDistance(Dets[0]) > schd.excitation) continue;
      }
      if (*it < *vec_it ) {
	Dets.push_back(*it);
	it++;
      }
      else if (*vec_it <*it && vec_it != SortedDets.end()) 
	vec_it++;
      else if (*vec_it < *it && vec_it == SortedDets.end()) {
	Dets.push_back(*it);
	it++;
      }
      else {
	vec_it++; it++;
      }
    }

    //now diagonalize the hamiltonian
    detChar.resize(norbs* Dets.size()); 
    MatrixXd X0(Dets.size(), 1); X0.setZero(Dets.size(),1); X0.block(0,0,ci.rows(),1) = 1.*ci; 
    MatrixXd diag(Dets.size(), 1); diag.setZero(diag.size(),1);
    if (mpigetrank() == 0) diag.block(0,0,ci.rows(),1)= 1.*diagOld;


    double estimatedCorrection = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+:estimatedCorrection)
    for (size_t k=SortedDets.size(); k<Dets.size() ; k++) {
      if (k % mpigetsize() != mpigetrank() ) continue;
      Dets[k].getRepArray(&detChar[norbs*k]);
      diag(k,0) = Energy(&detChar[norbs*k], Norbs, I1, I2, coreE);
      //double numerator = newDets[0][Dets[k]].first;
      //estimatedCorrection += numerator*numerator/(diag(k,0) - E0);
      if (k%1000000 == 0 && k!=0) cout <<"#"<< k<<"Hdiag out of "<<Dets.size()<<endl;     
    }
    newDets[0].clear();

#ifndef SERIAL
    //MPI_Allreduce(MPI_IN_PLACE, &estimatedCorrection, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &diag(0,0), diag.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    size_t intdim = (Dets.size()-SortedDets.size())*norbs;
    long  maxint = 26843540; //mpi cannot transfer more than these number of doubles
    long maxIter = intdim/maxint; 
    for (size_t ii=0; ii<maxIter; ii++) {
      MPI_Allreduce(MPI_IN_PLACE, &detChar[SortedDets.size()*norbs+ii*maxint], maxint, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Allreduce(MPI_IN_PLACE, &detChar[SortedDets.size()*norbs+maxIter*maxint], intdim-maxIter*maxint, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Allreduce(MPI_IN_PLACE, &detChar[SortedDets.size()*norbs], (Dets.size()-SortedDets.size())*norbs, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD);
#endif

    connections.resize(Dets.size());
    Helements.resize(Dets.size());
    orbDifference.resize(Dets.size());


    
    PopulateHelperLists(BetaN, AlphaNm1, Dets, ci.size());
    MakeHfromHelpers(BetaN, AlphaNm1, Dets, SortedDets.size(), connections, Helements,
		     &detChar[0], norbs, I1, I2, coreE, orbDifference, DoRDM);
     
    for (size_t i=SortedDets.size(); i<Dets.size(); i++)
      SortedDets.push_back(Dets[i]);
    std::sort(SortedDets.begin(), SortedDets.end());
 

    
    double prevE0 = E0;
    //Hmult H(&detChar[0], norbs, I1, I2, coreE);
    Hmult2 H(connections, Helements);
    
    E0 = davidson(H, X0, diag, 10, schd.davidsonTol, false);
    pout << format("# %4i  %10.2e  %10.2e   %14.8f  %10.2f") 
      %(iter) % epsilon1 % Dets.size() % E0 % (getTime()-startofCalc);
    pout << endl;

    ci.resize(Dets.size(),1); ci = 1.0*X0;
    diagOld.resize(Dets.size(),1); diagOld = 1.0*diag;

    
    if (abs(E0-prevE0) < schd.dE || iter == schd.epsilon1.size()-1)  {
      writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, Helements, E0, true, schd, BetaN, AlphaNm1);
      if (DoRDM) {	
	Helements.resize(0); BetaN.clear(); AlphaNm1.clear();
	EvaluateAndStoreRDM(connections, Dets, ci, orbDifference, nelec, schd);
      }
      break;
    }
    else {
      if (schd.io) writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, Helements, E0, false, schd, BetaN, AlphaNm1);
    }


  }
  return E0;

}



void CIPSIbasics::writeVariationalResult(int iter, MatrixXd& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
					 MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
					 double& E0, bool converged, schedule& schd,   
					 std::map<HalfDet, std::vector<int> >& BetaN, 
					 std::map<HalfDet, std::vector<int> >& AlphaNm1) {

#ifndef SERIAL
  boost::mpi::communicator world;
#endif

    pout << format("#Begin writing variational wf %29.2f\n")
      % (getTime()-startofCalc);

    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix.c_str(), world.rank() );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << iter <<Dets<<SortedDets;
    int diagrows = diag.rows();
    save << diagrows;
    for (int i=0; i<diag.rows(); i++)
      save << diag(i,0);
    for (int i=0; i<ci.rows(); i++)
      save << ci(i,0);
    save << E0;
    save << converged;
    save << connections<<Helements;
    save << BetaN<< AlphaNm1;
    ofs.close();

    pout << format("#End   writing variational wf %29.2f\n")
      % (getTime()-startofCalc);
}


void CIPSIbasics::readVariationalResult(int& iter, MatrixXd& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
					MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
					double& E0, bool& converged, schedule& schd,
					std::map<HalfDet, std::vector<int> >& BetaN, 
					std::map<HalfDet, std::vector<int> >& AlphaNm1) {


#ifndef SERIAL
  boost::mpi::communicator world;
#endif

    pout << format("#Begin reading variational wf %29.2f\n")
      % (getTime()-startofCalc);

    char file [5000];
    sprintf (file, "%s/%d-variational.bkp" , schd.prefix.c_str(), world.rank() );
    std::ifstream ifs(file, std::ios::binary);
    boost::archive::binary_iarchive load(ifs);
    
    load >> iter >> Dets >> SortedDets ;
    int diaglen;
    load >> diaglen;
    ci.resize(diaglen,1); diag.resize(diaglen,1);
    for (int i=0; i<diag.rows(); i++)
      load >> diag(i,0);
    for (int i=0; i<ci.rows(); i++)
      load >>  ci(i,0);
    load >> E0;
    if (schd.onlyperturbative) {ifs.close();return;}
    load >> converged;

    load >> connections >> Helements;
    load >> BetaN>> AlphaNm1;
    ifs.close();

    pout << format("#End   reading variational wf %29.2f\n")
      % (getTime()-startofCalc);
}


//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void CIPSIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<double>& numerator, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);

  double Energyd = Energy(closed, nclosed, int1, int2, coreE);
  

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;

    double integral = Hij_1Excite(closed[i],open[a],int1,int2, &closed[0], nclosed);
    

    if (fabs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);

      numerator.push_back(integral*ci1);
      energy.push_back(E);
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    if (true && (ints != I2hb.sameSpin.end() && ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital

	if (!(d.getocc(a) || d.getocc(b))) {
	  dets.push_back(d);
	  Determinant& di = *dets.rbegin();
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    

	  double sgn = 1.0;
	  di.parity(a, b, closed[i], closed[j], sgn);
	  numerator.push_back(it->first*sgn*ci1);

	  double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);	    
	  //double E = 0;
	  energy.push_back(E);

	}
      }
    }
  }
  return;
}


//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void CIPSIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);


  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;

    double integral = Hij_1Excite(closed[i],open[a],int1,int2, &closed[0], nclosed);

    if (fabs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    if (true && (ints != I2hb.sameSpin.end() && ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital

	if (!(d.getocc(a) || d.getocc(b))) {
	  dets.push_back(d);
	  Determinant& di = *dets.rbegin();
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    
	}
      }
    }
  }
  return;
}


  


//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void CIPSIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<double>& numerator1, vector<double>& numerator2, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);

  double Energyd = Energy(closed, nclosed, int1, int2, coreE);
  

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;

    double integral = Hij_1Excite(closed[i],open[a],int1,int2, &closed[0], nclosed);
    

    if (fabs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);

      numerator1.push_back(integral*ci1);
      numerator2.push_back(integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2));
      energy.push_back(E);
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    if (true && (ints != I2hb.sameSpin.end() && ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital

	if (!(d.getocc(a) || d.getocc(b))) {
	  dets.push_back(d);
	  Determinant& di = *dets.rbegin();
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    

	  double sgn = 1.0;
	  di.parity(a, b, closed[i], closed[j], sgn);

	  numerator1.push_back(it->first*sgn*ci1);
	  numerator2.push_back(it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2));

	  double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);	    
	  energy.push_back(E);

	}
      }
    }
  }
  return;
}

//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void CIPSIbasics::getDeterminants2Epsilon(Determinant& d, double epsilon, double epsilonLarge, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<double>& numerator1A, vector<double>& numerator2A, vector<double>& numerator1B, vector<double>& numerator2B, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);

  double Energyd = Energy(closed, nclosed, int1, int2, coreE);
  

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;

    double integral = Hij_1Excite(closed[i],open[a],int1,int2, &closed[0], nclosed);
    

    if (fabs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);

      numerator1A.push_back(integral*ci1);
      numerator2A.push_back(integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2));

      if (fabs(integral) >epsilonLarge) {
	numerator1B.push_back(integral*ci1);
	numerator2B.push_back(integral*integral*ci1*(ci1*Nmc/(Nmc-1)-ci2));
      }
      else {
	numerator1B.push_back(0.0); numerator2B.push_back(0.0);
      }

      energy.push_back(E);
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    if (true && (ints != I2hb.sameSpin.end() && ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital

	if (!(d.getocc(a) || d.getocc(b))) {
	  dets.push_back(d);
	  Determinant& di = *dets.rbegin();
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    

	  double sgn = 1.0;
	  di.parity(a, b, closed[i], closed[j], sgn);

	  numerator1A.push_back(it->first*sgn*ci1);
	  numerator2A.push_back(it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2));


	  if (fabs(it->first) >epsilonLarge) {
	    numerator1B.push_back(it->first*ci1*sgn);
	    numerator2B.push_back(it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2));
	  }
	  else {
	    numerator1B.push_back(0.0); numerator2B.push_back(0.0);
	  }
	  double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);	    
	  energy.push_back(E);

	}
      }
    }
  }
  return;
}


