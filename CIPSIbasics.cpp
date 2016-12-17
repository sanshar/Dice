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

void merge(Determinant *a, size_t low, size_t high, size_t mid, int* x, Determinant* c, int* cx)
{
  size_t i, j, k;
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

void mergesort(Determinant *a, int low, int high, int* x, Determinant* c, int* cx)
{
  int mid;
  if (low < high)
    {
      mid=(low+high)/2;
      mergesort(a,low,mid, x, c, cx);
      mergesort(a,mid+1,high, x, c, cx);
      merge(a,low,high,mid, x, c, cx);
    }
  return;
}

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

      std::vector<int> index(Psi1.size());
      for (int k=0; k<Psi1.size(); k++)
	index[k] = k;
      int* indexcopy = new int[Psi1.size()];
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


      std::vector<int> index(Psi1.size());
      for (int k=0; k<Psi1.size(); k++)
	index[k] = k;
      int* indexcopy = new int[Psi1.size()];
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

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    char psiArray[norbs]; vector<int> psiClosed(nelec,0), psiOpen(norbs-nelec,0);
    //char psiArray[norbs]; int psiOpen[nelec], psiClosed[norbs-nelec];
    double energyEN = 0.0;
    int num_thrds = omp_get_max_threads();


    std::vector<std::map<Determinant, pair<double,double> > > det_map(num_thrds);
    std::map<Determinant, pair<double,double> >::iterator det_it;

#pragma omp parallel 
    {
      size_t initSize = 100000;
      std::vector<Determinant> Psi1; std::vector<double>  numerator;
      std::vector<double>  det_energy;
      Psi1.reserve(initSize); numerator.reserve(initSize); det_energy.reserve(initSize);

      for (int i=0; i<Dets.size(); i++) {
	if (i%(omp_get_num_threads()) != omp_get_thread_num()) continue;
	CIPSIbasics::getDeterminants(Dets[i], abs(schd.epsilon2/ci(i,0)), ci(i,0), 0.0, I1, I2, I2HB, irrep, coreE, E0, Psi1, numerator, det_energy, schd,0, nelec);
	if (i%1000 == 0 && omp_get_thread_num()==0 && mpigetrank() == 0) cout <<"# "<<i<<endl;
      }


      if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "Before sort "<<getTime()-startofCalc<<endl;

      for (int i=0; i<Psi1.size(); i++) {      
	det_it = det_map[omp_get_thread_num()].lower_bound(Psi1[i]);
	if (det_it != det_map[omp_get_thread_num()].end() && Psi1[i] == det_it->first) {
	  det_it->second.first += numerator[i];
	}
	else {
	  det_map[omp_get_thread_num()].insert(det_it, map<Determinant, pair<double, double> >::value_type(Psi1[i], pair<double, double>(numerator[i], det_energy[i]) ));
	}

      }


      if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "Unique determinants "<<getTime()-startofCalc<<endl;

      for (int level=0; level<ceil(log2(omp_get_num_threads())); level++) {
#pragma omp barrier
	if (omp_get_thread_num()%ipow(2,level+1) == 0 && omp_get_thread_num() + ipow(2,level) < omp_get_num_threads() ) {

	  int other_thrd = omp_get_thread_num()+ipow(2,level);

	  for (map<Determinant, pair<double, double> >::iterator it=det_map[other_thrd].begin(); it!=det_map[other_thrd].end(); ++it)  {
	    map<Determinant, pair<double, double> >::iterator det_it = det_map[omp_get_thread_num()].lower_bound( it->first );
	    
	    if(det_it != det_map[omp_get_thread_num()].end() && det_it->first == it->first)  
	      det_it->second.first += it->second.first;
	    else
	      det_map[omp_get_thread_num()].insert(det_it, map<Determinant, pair<double, double> >::value_type(it->first, pair<double, double>(it->second.first, it->second.second) ) );
	  }
	  det_map[other_thrd].clear();
	}	  

      }

      if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "Merging "<<getTime()-startofCalc<<endl;

    }
    det_map.resize(1);




    double totalPT=0.0;
#pragma omp parallel 
    {
      double PTEnergy = 0.0;
      map<Determinant, pair<double, double> >::iterator map_it = det_map[0].begin();

      vector<Determinant>::iterator vec_it = SortedDets.begin();
      size_t startIter = omp_get_thread_num() * (det_map[0].size()/omp_get_num_threads()) ;
      size_t endIter = startIter +  (det_map[0].size()/omp_get_num_threads()) ;
      if (omp_get_thread_num() == num_thrds-1) endIter = det_map[0].size();

      size_t cnt = 0;
      while(map_it != det_map[0].end() ) {
	if (cnt <startIter) {cnt++; map_it++;}
	else if (cnt >=endIter) break;
	else if ( map_it->first < *vec_it ) {
	  PTEnergy += map_it->second.first*map_it->second.first/(E0-map_it->second.second); 
	  map_it++;
	  cnt++;
	}
	else if (*vec_it < map_it->first && vec_it != SortedDets.end())
	  vec_it++;
	else if (*vec_it < map_it->first && vec_it == SortedDets.end()) {
	  PTEnergy += map_it->second.first*map_it->second.first/(E0-map_it->second.second); 
	  map_it++;
	  cnt++;
	}
	else {
	  vec_it++; map_it++;cnt++;
	}
      }

#pragma omp critical
      {
	totalPT += PTEnergy;
      }
    }

    if (mpigetrank() == 0) cout << "Done energy "<<totalPT<<"  "<<getTime()-startofCalc<<endl;
    return totalPT;
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
		      double& coreE) {

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
	      double hij = Hij(Dets[J], Dets[K], I1, I2, coreE);
	      Helements[K].push_back(hij);
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
	      double hij = Hij(Dets[J], Dets[K], I1, I2, coreE);
	      //double hij = Hij(&detChar[norbs*J], &detChar[norbs*K], Norbs, I1, I2, coreE);
	      Helements[K].push_back(hij);
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
				  , int nelec) {

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
  if (mpigetrank() != 0) {connections.resize(0); Helements.resize(0);}


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

      int* indices = new int[newDets[omp_get_thread_num()].size()];
      int* ic = new int[newDets[omp_get_thread_num()].size()];
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
      else {
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


    
    if (true) {
      PopulateHelperLists(BetaN, AlphaNm1, Dets, ci.size());
      MakeHfromHelpers(BetaN, AlphaNm1, Dets, SortedDets.size(), connections, Helements,
		       &detChar[0], norbs, I1, I2, coreE);
    }
    //if (Dets.size() > 10000000) {
    else if (false){
      size_t oldDetSize = SortedDets.size();
      map<Determinant, int> SortedDets;
      cout << "about to sort"<<endl;
      for (int i=0; i<oldDetSize; i++)
	SortedDets[Dets[i]] = i;
      cout << "update connections"<<endl;
      updateConnections(Dets, SortedDets, norbs, I1, I2, coreE, &detChar[0], connections, Helements);
    //update connetions
    }
    else
    {
#pragma omp parallel for schedule(dynamic)
      for (size_t i=0; i<Dets.size() ; i++) {
	if (i%mpigetsize() != mpigetrank()) continue;
	
	for (size_t j=max(SortedDets.size(),i); j<Dets.size(); j++) {
	  if (Dets[i].connected(Dets[j])) {
	    double hij = Hij(&detChar[norbs*i], &detChar[norbs*j], Norbs, I1, I2, coreE);
	    
	    if (abs(hij) > 1.e-10) {
	      connections[i].push_back(j);
	      Helements[i].push_back(hij);
	    }
          }
	}

        if (i%1000000 == 0 && i!=0) cout <<"#"<< i<<" Hij out of "<<Dets.size()<<endl;     

        //if (i%10000) cout << i<<"  "<<Dets.size()<<endl;
      }
    } 
     
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

    
    if (abs(E0-prevE0) < schd.dE)  {
      writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, Helements, E0, true, schd, BetaN, AlphaNm1);
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


