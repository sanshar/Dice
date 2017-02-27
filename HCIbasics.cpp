#include "Determinants.h"
#include "HCIbasics.h"
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

void RemoveDuplicates(vector<Determinant>& Det) {
  if (Det.size() <= 1) return;
  std::vector<Determinant>& Detcopy = Det;
  size_t uniqueSize = 0;
  for (size_t i=1; i <Detcopy.size(); i++) {
    if (!(Detcopy[i] == Detcopy[i-1])) {
      uniqueSize++;
      Det[uniqueSize] = Detcopy[i];
    }
  }
  Det.resize(uniqueSize+1);
}

int partition(Determinant* A, int p,int q, double* pNum, double* Energy, vector<double>* det_energy=NULL, vector<bool>* present=NULL)
{
  Determinant x= A[p];
  int i=p;
  int j;

  for(j=p+1; j<q; j++)
    {
      if(A[j]<x || A[j] == x)
        {
	  i=i+1;
	  swap(A[i],A[j]);
	  swap(pNum[i], pNum[j]);
	  swap(Energy[i], Energy[j]);

	  if (present != NULL) {
	    bool bkp = present->operator[](j);
	    present->operator[](j) = present->operator[](i);
	    present->operator[](i) = bkp;
	  }
	  if (det_energy != NULL)
	    swap(det_energy->operator[](i), det_energy->operator[](j));
        }

    }

  swap(A[i],A[p]);
  swap(pNum[i], pNum[p]);
  swap(Energy[i], Energy[p]);
  if (present != NULL) {
    bool bkp = present->operator[](p);
    present->operator[](p) = present->operator[](i);
    present->operator[](i) = bkp;
  }

  if (det_energy != NULL)
    swap(det_energy->operator[](i), det_energy->operator[](p));
  return i;
}

int partitionAll(Determinant* A, int p,int q, double* pNum, double* Energy, vector<int>* var_indices, vector<int>* orbDifference, vector<double>* det_energy=NULL, vector<bool>* present=NULL)
{
  Determinant x= A[p];
  int i=p;
  int j;

  for(j=p+1; j<q; j++)
    {
      if(A[j]<x || A[j] == x)
        {
	  i=i+1;
	  swap(A[i],A[j]);
	  swap(pNum[i], pNum[j]);
	  swap(Energy[i], Energy[j]);
	  swap(var_indices[i], var_indices[j]);
	  swap(orbDifference[i], orbDifference[j]);

	  if (present != NULL) {
	    bool bkp = present->operator[](j);
	    present->operator[](j) = present->operator[](i);
	    present->operator[](i) = bkp;
	  }
	  if (det_energy != NULL)
	    swap(det_energy->operator[](i), det_energy->operator[](j));
        }

    }

  swap(A[i],A[p]);
  swap(pNum[i], pNum[p]);
  swap(Energy[i], Energy[p]);
  swap(var_indices[i], var_indices[p]);
  swap(orbDifference[i], orbDifference[p]);
  if (present != NULL) {
    bool bkp = present->operator[](p);
    present->operator[](p) = present->operator[](i);
    present->operator[](i) = bkp;
  }

  if (det_energy != NULL)
    swap(det_energy->operator[](i), det_energy->operator[](p));
  return i;
}

void quickSort(Determinant* A, int p,int q, double* pNum, double* Energy, vector<double>* det_energy=NULL, vector<bool>* present=NULL)
{
  int r;
  if(p<q)
    {
      r=partition(A,p,q, pNum, Energy, det_energy, present);
      quickSort(A,p,r, pNum, Energy, det_energy, present);  
      quickSort(A,r+1,q, pNum, Energy, det_energy, present);
    }
}

void quickSortAll(Determinant* A, int p,int q, double* pNum, double* Energy, vector<int>* var_indices, vector<int>* orbDifference, vector<double>* det_energy=NULL, vector<bool>* present=NULL)
{
  int r;
  if(p<q)
    {
      r=partitionAll(A,p,q, pNum, Energy, var_indices, orbDifference, det_energy, present);
      quickSortAll(A,p,r, pNum, Energy, var_indices, orbDifference, det_energy, present);  
      quickSortAll(A,r+1,q, pNum, Energy, var_indices, orbDifference, det_energy, present);
    }
}



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
      ar & Det & Num & Energy & var_indices & orbDifference;
  }

public:
  boost::shared_ptr<vector<Determinant> > Det;
  boost::shared_ptr<vector<double> > Num;
  boost::shared_ptr<vector<double> > Energy;
  boost::shared_ptr<vector<vector<int> > > var_indices;
  boost::shared_ptr<vector<vector<int> > > orbDifference;
  bool extra_info; // whether to use var_indices, orbDifference
  
  StitchDEH() {
    Det = boost::shared_ptr<vector<Determinant> > (new vector<Determinant>() );
    Num = boost::shared_ptr<vector<double> > (new vector<double>() );
    Energy = boost::shared_ptr<vector<double> > (new vector<double>() );
    extra_info = false;
    var_indices = boost::shared_ptr<vector<vector<int> > > (new vector<vector<int> >() );
    orbDifference = boost::shared_ptr<vector<vector<int> > > (new vector<vector<int> >() );
  }

  StitchDEH(boost::shared_ptr<vector<Determinant> >pD, 
	    boost::shared_ptr<vector<double> >pNum, 
	    boost::shared_ptr<vector<double> >pE, 
	    boost::shared_ptr<vector<vector<int> > >pvar, 
	    boost::shared_ptr<vector<vector<int> > >porb)
  : Det(pD), Num(pNum), Energy(pE), var_indices(pvar), orbDifference(porb)
    {
      extra_info=true;
    };

  StitchDEH(boost::shared_ptr<vector<Determinant> >pD, 
	    boost::shared_ptr<vector<double> >pNum, 
	    boost::shared_ptr<vector<double> >pE)
  : Det(pD), Num(pNum), Energy(pE)
    {
      extra_info=false;
    };


  void QuickSortAndRemoveDuplicates() {
    if (extra_info) {
      quickSortAll(&(Det->operator[](0)), 0, Det->size(), &(Num->operator[](0)), &(Energy->operator[](0)), &(var_indices->operator[](0)), &(orbDifference->operator[](0)));
    } else {
      quickSort(&(Det->operator[](0)), 0, Det->size(), &(Num->operator[](0)), &(Energy->operator[](0)));
    }

    //if (Det->size() == 1) return;
    if (Det->size() <= 1) return;
    
    std::vector<Determinant>& Detcopy = *Det;
    std::vector<double>& Numcopy = *Num;
    std::vector<double>& Ecopy = *Energy;
  //if (extra_info) {
      std::vector<std::vector<int> >& Vcopy = *var_indices;
      std::vector<std::vector<int> >& Ocopy = *orbDifference;
  //}
    size_t uniqueSize = 0;
    for (size_t i=1; i <Detcopy.size(); i++) {
      if (!(Detcopy[i] == Detcopy[i-1])) {
	uniqueSize++;
	Det->operator[](uniqueSize) = Detcopy[i];
	Num->operator[](uniqueSize) = Numcopy[i];
	Energy->operator[](uniqueSize) = Ecopy[i];
        if (extra_info) {
	  var_indices->operator[](uniqueSize) = Vcopy[i];
	  orbDifference->operator[](uniqueSize) = Ocopy[i];
        }
      }
      else { // Same det, so combine
	Num->operator[](uniqueSize) += Numcopy[i];
        if (extra_info) {
          for (size_t k=0; k<(*var_indices)[i].size(); k++) {
    	    (*var_indices)[uniqueSize].push_back((*var_indices)[i][k]);
    	    (*orbDifference)[uniqueSize].push_back((*orbDifference)[i][k]);
          }
        }
      }
    }
    Det->resize(uniqueSize+1);
    Num->resize(uniqueSize+1);
    Energy->resize(uniqueSize+1);
    if (extra_info) {
      var_indices->resize(uniqueSize+1);
      orbDifference->resize(uniqueSize+1);
    }
    
  }

/* Commenting out because it's not used -AAH
  void MergeSortAndRemoveDuplicates() {
    std::vector<Determinant> Detcopy = *Det;

    long* detIndex =  new long[Detcopy.size()];
    long* detIndexcopy = new long[Detcopy.size()];
    for (size_t i=0; i<Detcopy.size(); i++)
      detIndex[i] = i;      
    mergesort(&Detcopy[0], 0, Detcopy.size()-1, detIndex, &( Det->operator[](0)), detIndexcopy);
    delete [] detIndexcopy;

    //if (Det->size() == 1) return;
    if (Det->size() <= 1) return;
    std::vector<double> Numcopy = *Num;
    std::vector<double> Ecopy = *Energy;
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

    delete [] detIndex;
  }
*/

  void RemoveDetsPresentIn(std::vector<Determinant>& SortedDets) {
      vector<Determinant>::iterator vec_it = SortedDets.begin();
      std::vector<Determinant>& Detcopy = *Det;
      std::vector<double>& Numcopy = *Num;
      std::vector<double>& Ecopy = *Energy;
    //if (extra_info) {
        std::vector<std::vector<int> >& Vcopy = *var_indices;
        std::vector<std::vector<int> >& Ocopy = *orbDifference;
    //}

      size_t uniqueSize = 0;
      for (size_t i=0; i<Detcopy.size();) {
	if (Detcopy[i] < *vec_it) {
	  Det->operator[](uniqueSize) = Detcopy[i];
	  Num->operator[](uniqueSize) = Numcopy[i];
	  Energy->operator[](uniqueSize) = Ecopy[i];
          if (extra_info) {
	    var_indices->operator[](uniqueSize) = Vcopy[i];
	    orbDifference->operator[](uniqueSize) = Ocopy[i];
          }
	  i++; uniqueSize++;
	}
	else if (*vec_it < Detcopy[i] && vec_it != SortedDets.end())
	  vec_it ++;
	else if (*vec_it < Detcopy[i] && vec_it == SortedDets.end()) {
	  Det->operator[](uniqueSize) = Detcopy[i];
	  Num->operator[](uniqueSize) = Numcopy[i];
	  Energy->operator[](uniqueSize) = Ecopy[i];
          if (extra_info) {
	    var_indices->operator[](uniqueSize) = Vcopy[i];
	    orbDifference->operator[](uniqueSize) = Ocopy[i];
          }
	  i++; uniqueSize++;
	}
	else {
	  vec_it++; i++;
	}
      }
      Det->resize(uniqueSize); Num->resize(uniqueSize); Energy->resize(uniqueSize);
      if (extra_info) {
        var_indices->resize(uniqueSize); orbDifference->resize(uniqueSize);
      }
  }

  void RemoveDuplicates() {
      std::vector<Determinant>& Detcopy = *Det;
      std::vector<double>& Numcopy = *Num;
      std::vector<double>& Ecopy = *Energy;
    //if (extra_info) {
        std::vector<std::vector<int> >& Vcopy = *var_indices;
        std::vector<std::vector<int> >& Ocopy = *orbDifference;
    //}

      if (Det->size() <= 1) return;
      //if (Det->size() == 1) return;
      size_t uniqueSize = 0;
      for (size_t i=1; i <Detcopy.size(); i++) {
	if (!(Detcopy[i] == Detcopy[i-1])) {
	  uniqueSize++;
	  Det->operator[](uniqueSize) = Detcopy[i];
	  Num->operator[](uniqueSize) = Numcopy[i];
	  Energy->operator[](uniqueSize) = Ecopy[i];
          if (extra_info) {
	    var_indices->operator[](uniqueSize) = Vcopy[i];
  	    orbDifference->operator[](uniqueSize) = Ocopy[i];
	  }
	}
	else {// Same det, so combine
	  Num->operator[](uniqueSize) += Numcopy[i];
          if (extra_info) {
            for (size_t k=0; k<(*var_indices)[i].size(); k++) {
    	      (*var_indices)[uniqueSize].push_back((*var_indices)[i][k]);
    	      (*orbDifference)[uniqueSize].push_back((*orbDifference)[i][k]);
            }
          }
        }
      }
      Det->resize(uniqueSize+1);
      Num->resize(uniqueSize+1);
      Energy->resize(uniqueSize+1);
      if (extra_info) {
        var_indices->resize(uniqueSize+1);
        orbDifference->resize(uniqueSize+1);
      }
  }

  void deepCopy(const StitchDEH& s) {
    *Det = *(s.Det);
    *Num = *(s.Num);
    *Energy = *(s.Energy);
    if (extra_info) {
      *var_indices = *(s.var_indices);
      *orbDifference = *(s.orbDifference);
    }
  }
  
  void operator=(const StitchDEH& s) {
    Det = s.Det;
    Num = s.Num;
    Energy = s.Energy;
    if (extra_info) {
      var_indices = s.var_indices;
      orbDifference = s.orbDifference;
    }
  }

  void clear() {
    Det->clear();
    Num->clear();
    Energy->clear();
    if (extra_info) {
      var_indices->clear();
      orbDifference->clear();
    }
  }

  void merge(const StitchDEH& s) {
    // Merges with disjoint set
    std::vector<Determinant> Detcopy = *Det;
    std::vector<double> Numcopy = *Num;
    std::vector<double> Ecopy = *Energy;
  //if (extra_info) {
      std::vector<std::vector<int> > Vcopy = *var_indices;
      std::vector<std::vector<int> > Ocopy = *orbDifference;
  //}
    
    Det->resize(Detcopy.size()+s.Det->size());
    Num->resize(Numcopy.size()+s.Det->size());
    Energy->resize(Ecopy.size()+s.Energy->size());
    if (extra_info) {
      var_indices->resize(Vcopy.size()+s.var_indices->size());
      orbDifference->resize(Ocopy.size()+s.orbDifference->size());
    }

    //cout << Det->size()<<"  "<<Detcopy.size()<<"  "<<s.Det->size()<<endl;
    size_t j = 0, k=0,l=0;
    while (j<Detcopy.size() && k <s.Det->size()) {
      if (Detcopy.operator[](j) < s.Det->operator[](k)) {
	Det->operator[](l) = Detcopy.operator[](j);
	Num->operator[](l) = Numcopy.operator[](j);
	Energy->operator[](l) = Ecopy.operator[](j);
        if (extra_info) {
	  var_indices->operator[](l) = Vcopy.operator[](j);
	  orbDifference->operator[](l) = Ocopy.operator[](j);
        }
	j++; l++;
      }
      else {
	Det->operator[](l) = s.Det->operator[](k);
	Num->operator[](l) = s.Num->operator[](k);
	Energy->operator[](l) = s.Energy->operator[](k);
        if (extra_info) {
	  var_indices->operator[](l) = s.var_indices->operator[](k);
	  orbDifference->operator[](l) = s.orbDifference->operator[](k);
        }
	k++;l++;
      }
    }
    while (j<Detcopy.size()) {
      Det->operator[](l) = Detcopy.operator[](j);
      Num->operator[](l) = Numcopy.operator[](j);
      Energy->operator[](l) = Ecopy.operator[](j);
      if (extra_info) {
        var_indices->operator[](l) = Vcopy.operator[](j);
        orbDifference->operator[](l) = Ocopy.operator[](j);
      }
      j++; l++;
    }
    while (k<s.Det->size()) {
      //cout << l<<"  "<<k<<"  "<<Det->size()<<"  "<<s.Det->size()<<"  "<<j<<"  "<<Detcopy.size()<<endl;
      Det->operator[](l) = s.Det->operator[](k);
      Num->operator[](l) = s.Num->operator[](k);
      Energy->operator[](l) = s.Energy->operator[](k);
      if (extra_info) {
        var_indices->operator[](l) = s.var_indices->operator[](k);
        orbDifference->operator[](l) = s.orbDifference->operator[](k);
      }
      k++;l++;
    }
    
  } // end merge

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
int HCIbasics::sample_round(MatrixXd& ci, double eps, std::vector<int>& Sample1, std::vector<double>& newWts){
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


void HCIbasics::EvaluateAndStoreRDM(vector<vector<int> >& connections, vector<Determinant>& Dets, MatrixXd& ci,
				      vector<vector<size_t> >& orbDifference, int nelec, schedule& schd, int root, MatrixXd& twoRDM) {
  boost::mpi::communicator world;

  size_t norbs = Dets[0].norbs;

  //RDM(i,j,k,l) = a_i^\dag a_j^\dag a_l a_k
  //also i>=j and k>=l
//MatrixXd twoRDM(norbs*(norbs+1)/2, norbs*(norbs+1)/2);
  twoRDM *= 0.0;
  int num_thrds = omp_get_max_threads();

  //#pragma omp parallel for schedule(dynamic)
  for (int i=0; i<Dets.size(); i++) {
    if ((i/num_thrds)%world.size() != world.rank()) continue;

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

  MPI_Allreduce(MPI_IN_PLACE, &twoRDM(0,0), twoRDM.rows()*twoRDM.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

}



void HCIbasics::ComputeEnergyFromRDM(int norbs, int nelec, oneInt& I1, twoInt& I2, double coreE, MatrixXd& twoRDM) {

  //RDM(i,j,k,l) = a_i^\dag a_j^\dag a_l a_k
  //also i>=j and k>=l
  double energy = coreE;
  double onebody = 0.0;
  double twobody = 0.0;
  //if (mpigetrank() == 0)  cout << "Core energy= " << energy << endl; 

  MatrixXd oneRDM = MatrixXd::Zero(norbs, norbs);
  for (int p=0; p<norbs; p++)
  for (int q=0; q<norbs; q++)
    for (int r=0; r<norbs; r++) {
      int P = max(p,r), R1 = min(p,r);
      int Q = max(q,r), R2 = min(q,r);
      int sgn = 1;
      if (P != p)  sgn *= -1;
      if (Q != q)  sgn *= -1;

      oneRDM(p,q) += sgn*twoRDM(P*(P+1)/2+R1,Q*(Q+1)/2+R2)/(nelec-1);
    }
    
  for (int p=0; p<norbs; p++)
  for (int q=0; q<norbs; q++)
    onebody += I1(p, q)*oneRDM(p,q);

  for (int p=0; p<norbs; p++){
  for (int q=0; q<norbs; q++){
  for (int r=0; r<norbs; r++){
  for (int s=0; s<norbs; s++){
      //if (p%2 != r%2 || q%2 != s%2)  continue; // This line is not necessary
      int P = max(p,q), Q = min(p,q);
      int R = max(r,s), S = min(r,s);
      int sgn = 1;
      if (P != p)  sgn *= -1;
      if (R != r)  sgn *= -1;
      twobody += sgn * 0.5 * twoRDM(P*(P+1)/2+Q, R*(R+1)/2+S) * I2(p,r,q,s); // 2-body term
  }
  }
  }
  }
  
  //if (mpigetrank() == 0)  cout << "One-body from 2RDM: " << onebody << endl;
  //if (mpigetrank() == 0)  cout << "Two-body from 2RDM: " << twobody << endl;

  energy += onebody + twobody;

  if (mpigetrank() == 0)  cout << "E from 2RDM: " << energy << endl;

}



void HCIbasics::printRDM(int norbs, schedule& schd, int root, MatrixXd& twoRDM) {

  boost::mpi::communicator world;

  //RDM(i,j,k,l) = a_i^\dag a_j^\dag a_l a_k
  //also i>=j and k>=l
  int num_thrds = omp_get_max_threads();

  int nSpatOrbs = norbs/2;
  MatrixXd s2RDM(nSpatOrbs*nSpatOrbs, nSpatOrbs*nSpatOrbs);
  s2RDM *= 0.0;


  if(mpigetrank() == 0) {
  char file [5000];
  sprintf (file, "%s/spatialRDM.%d.%d.txt" , schd.prefix.c_str(), root, root );
  std::ofstream ofs(file, std::ios::out);
  ofs << nSpatOrbs<<endl;

#pragma omp parallel for schedule(static)
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

  for (int n1=0; n1<nSpatOrbs; n1++)
  for (int n2=0; n2<nSpatOrbs; n2++)
  for (int n3=0; n3<nSpatOrbs; n3++)
  for (int n4=0; n4<nSpatOrbs; n4++)
  {
    if (fabs(s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4))  > 1.e-6)
      ofs << str(boost::format("%3d   %3d   %3d   %3d   %10.8g\n") % n1 % n2 % n3 % n4 % s2RDM(n1*nSpatOrbs+n2, n3*nSpatOrbs+n4));
  }


  {
    char file [5000];
    sprintf (file, "%s/%d-spinRDM.bkp" , schd.prefix.c_str(), root );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << twoRDM;
  }
  {
    char file [5000];
    sprintf (file, "%s/%d-spatialRDM.bkp" , schd.prefix.c_str(), root );
    std::ofstream ofs(file, std::ios::binary);
    boost::archive::binary_oarchive save(ofs);
    save << s2RDM;
  }
  }
}

void HCIbasics::setUpAliasMethod(MatrixXd& ci, double& cumulative, std::vector<int>& alias, std::vector<double>& prob) {
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

int HCIbasics::sample_N2_alias(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts, std::vector<int>& alias, std::vector<double>& prob) {

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

int HCIbasics::sample_N2(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts){
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

int HCIbasics::sample_N(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts){
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


void HCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
								    twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, int root) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  std::ofstream ofs;
  if (root == 0)
    ofs.open(file, std::ofstream::out);
  else
    ofs.open(file, std::ofstream::app);

  double epsilon2 = schd.epsilon2;
  schd.epsilon2 = schd.epsilon2Large;
  double EptLarge = DoPerturbativeDeterministic(Dets, ci, E0, I1, I2, I2HB, irrep, schd, coreE, nelec);

  schd.epsilon2 = epsilon2;

  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  int niter = schd.nPTiter;
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
    vector<bool> present;
    std::vector<double>  det_energy;
    for (int i=0; i<distinctSample; i++) {
      int I = Sample1[i];
      HCIbasics::getDeterminants2Epsilon(Dets[I], abs(schd.epsilon2/ci(I,0)), abs(schd.epsilon2Large/ci(I,0)), wts1[i], ci(I,0), I1, I2, I2HB, irrep, coreE, E0, Psi1, numerator1A, numerator2A, present, det_energy, schd, Nmc, nelec);
    }


    quickSort( &(Psi1[0]), 0, Psi1.size(), &numerator1A[0], &numerator2A[0], &det_energy, &present);
    
    double currentNum1A=0., currentNum2A=0.;
    double currentNum1B=0., currentNum2B=0.;
    vector<Determinant>::iterator vec_it = SortedDets.begin();
    double energyEN = 0.0, energyENLargeEps = 0.0;
    
    for (int i=0;i<Psi1.size();) {
      if (Psi1[i] < *vec_it) {
	currentNum1A += numerator1A[i];
	currentNum2A += numerator2A[i];
	if (present[i]) {
	  currentNum1B += numerator1A[i];
	  currentNum2B += numerator2A[i];
	}
	
	if ( i == Psi1.size()-1) {
	  energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	  energyENLargeEps += (currentNum1B*currentNum1B*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
	}
	else if (!(Psi1[i] == Psi1[i+1])) {
	  energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	  energyENLargeEps += (currentNum1B*currentNum1B*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
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
	currentNum1A += numerator1A[i];
	currentNum2A += numerator2A[i];
	if (present[i]) {
	  currentNum1B += numerator1A[i];
	  currentNum2B += numerator2A[i];
	}
	
	if ( i == Psi1.size()-1) {
	  energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	  energyENLargeEps += (currentNum1B*currentNum1B*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
	}
	else if (!(Psi1[i] == Psi1[i+1])) {
	  energyEN += (currentNum1A*currentNum1A*Nmc/(Nmc-1) - currentNum2A)/(det_energy[i] - E0);
	  energyENLargeEps += (currentNum1B*currentNum1B*Nmc/(Nmc-1) - currentNum2B)/(det_energy[i] - E0);
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
	std::cout << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i %4i") 
	  %(currentIter) % (E0-energyEN+energyENLargeEps+EptLarge) % ("Root") % root % (E0+AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	cout << endl;
      }
      else {
	AvgenergyEN += -energyEN+energyENLargeEps+EptLarge; currentIter++;
	ofs << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i %4i") 
	  %(currentIter) % (E0-energyEN+energyENLargeEps+EptLarge) % ("Root") % root % (E0+AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	ofs << endl;
	
      }
    }
  }
  ofs.close();
  
}

void HCIbasics::DoPerturbativeStochastic2SingleList(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
						      twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, int root) {

  boost::mpi::communicator world;
  char file [5000];
  sprintf (file, "output-%d.bkp" , world.rank() );
  //std::ofstream ofs(file);
  std::ofstream ofs;
  if (root == 0)
    ofs.open(file, std::ofstream::out);
  else
    ofs.open(file, std::ofstream::app);

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = schd.nPTiter;
  //int niter = 1000000;
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
       HCIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], ci(I,0), I1, I2, I2HB, irrep, coreE, E0, Psi1, numerator1, numerator2, det_energy, schd, Nmc, nelec);
      }


      quickSort( &(Psi1[0]), 0, Psi1.size(), &numerator1[0], &numerator2[0], &det_energy);


      double currentNum1=0., currentNum2=0.;
      vector<Determinant>::iterator vec_it = SortedDets.begin();
      double energyEN = 0.0;

      for (int i=0;i<Psi1.size();) {
	if (Psi1[i] < *vec_it) {
	  currentNum1 += numerator1[i];
	  currentNum2 += numerator2[i];
	  if ( i == Psi1.size()-1) 
	    energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) - currentNum2)/(det_energy[i] - E0);
	  else if (!(Psi1[i] == Psi1[i+1])) {
	    energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) - currentNum2)/(det_energy[i] - E0);
	    currentNum1 = 0.;
	    currentNum2 = 0.;
	  }
	  i++;
	}
	else if (*vec_it <Psi1[i] && vec_it != SortedDets.end())
	  vec_it++;
	else if (*vec_it <Psi1[i] && vec_it == SortedDets.end()) {
	  currentNum1 += numerator1[i];
	  currentNum2 += numerator2[i];
	  if ( i == Psi1.size()-1) 
	    energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) - currentNum2)/(det_energy[i] - E0);
	  else if (!(Psi1[i] == Psi1[i+1])) {
	    energyEN += (currentNum1*currentNum1*Nmc/(Nmc-1) - currentNum2)/(det_energy[i] - E0);
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
	  std::cout << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % ("Root") % root % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  cout << endl;
	}
	else {
	  AvgenergyEN += energyEN; currentIter++;
	  ofs << format("%6i  %14.8f  %s%i %14.8f   %10.2f  %10i %4i") 
	    %(currentIter) % (E0-energyEN) % ("Root") % root % (E0-AvgenergyEN/currentIter) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	  ofs << endl;
	  
	}
      /*
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
      */
      }
    }
    ofs.close();

}

double HCIbasics::DoPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					      twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, bool appendPsi1ToPsi0) {

  boost::mpi::communicator world;
  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  char psiArray[norbs]; vector<int> psiClosed(nelec,0), psiOpen(norbs-nelec,0);
  //char psiArray[norbs]; int psiOpen[nelec], psiClosed[norbs-nelec];
  double energyEN = 0.0;
  int num_thrds = omp_get_max_threads();
  
  
  std::vector<StitchDEH> uniqueDEH(num_thrds);
  double totalPT = 0.0;
#pragma omp parallel 
  {
    for (int i=0; i<Dets.size(); i++) {
      if (i%(omp_get_num_threads()) != omp_get_thread_num()) {continue;}
      HCIbasics::getDeterminants(Dets[i], abs(schd.epsilon2/ci(i,0)), ci(i,0), 0.0, 
				   I1, I2, I2HB, irrep, coreE, E0, 
				   *uniqueDEH[omp_get_thread_num()].Det, 
				   *uniqueDEH[omp_get_thread_num()].Num, 
				   *uniqueDEH[omp_get_thread_num()].Energy, 
				   schd,0, nelec);
      if (i%100000 == 0 && omp_get_thread_num()==0 && mpigetrank() == 0) cout <<"# "<<i<<endl;
    }
    
    
    if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#Before sort "<<getTime()-startofCalc<<endl;
    
    //uniqueDEH[omp_get_thread_num()].MergeSortAndRemoveDuplicates();
    uniqueDEH[omp_get_thread_num()].QuickSortAndRemoveDuplicates();
    uniqueDEH[omp_get_thread_num()].RemoveDetsPresentIn(SortedDets);

    if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#Unique determinants "<<getTime()-startofCalc<<"  "<<endl;
    
#pragma omp barrier 
    if (omp_get_thread_num() == 0) {
      for (int thrd=1; thrd<num_thrds; thrd++) {
	uniqueDEH[0].merge(uniqueDEH[thrd]);
	uniqueDEH[thrd].clear();
	uniqueDEH[0].RemoveDuplicates();
      }
    }
  }

  if (mpigetrank() == 0 ) cout << "#Before mpi split "<<getTime()-startofCalc<<"  "<<uniqueDEH[0].Det->size()<<endl;
  uniqueDEH.resize(1);
  
  
  vector<Determinant>& uniqueDets = *uniqueDEH[0].Det;
  vector<double>& uniqueNumerator = *uniqueDEH[0].Num;
  vector<double>& uniqueEnergy = *uniqueDEH[0].Energy;
  totalPT=0.0;

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


  double PT3 = 0.0;

  double finalE = 0.;
  mpi::all_reduce(world, totalPT, finalE, std::plus<double>());
  
  if (mpigetrank() == 0) cout << "#Done energy "<<E0+finalE<<"  "<<getTime()-startofCalc<<endl;

  if (0) {
    //Calculate the third order energy
    //this only works for one mpi process calculations
    size_t orbDiff;
    double norm = 0.0;
    double diag1 = 0.0; //<psi1|H0|psi1>
#pragma omp parallel for reduction(+:norm, PT3,diag1)
    for (int i=0; i<uniqueDets.size(); i++) {
      //cout << uniqueNumerator[i]/(E0-uniqueEnergy[i])<<"  "<<(*uniqueDEH[0].Det)[i]<<endl;
      norm += uniqueNumerator[i]*uniqueNumerator[i]/(E0-uniqueEnergy[i])/(E0-uniqueEnergy[i]);
      diag1 += uniqueEnergy[i]*pow(uniqueNumerator[i]/(E0-uniqueEnergy[i]), 2);
      for (int j=i+1; j<uniqueDets.size(); j++)
	if (uniqueDets[i].connected(uniqueDets[j])) {
	  double hij = Hij(uniqueDets[i], uniqueDets[j], I1, I2, coreE, orbDiff);
	  PT3 += 2.*hij*uniqueNumerator[i]*uniqueNumerator[j]/(E0-uniqueEnergy[i])/(E0-uniqueEnergy[j]);
	}
    }
    cout << "PT3 "<<PT3<<"  "<<norm<<"  "<<PT3+E0+finalE<<"  "<<(E0+2*finalE+PT3+diag1)/(1+norm)<<endl;
    PT3 = 0;
  }

  //Calculate Psi1 and then add it to Psi10
  if (appendPsi1ToPsi0) {
#ifndef SERIAL
    for (int level = 0; level <ceil(log2(mpigetsize())); level++) {
      
      if (mpigetrank()%ipow(2, level+1) == 0 && mpigetrank() + ipow(2, level) < mpigetsize()) {
	StitchDEH recvDEH;	
	int getproc = mpigetrank()+ipow(2,level);
	world.recv(getproc, mpigetsize()*level+getproc, recvDEH);
	uniqueDEH[0].merge(recvDEH);
	uniqueDEH[0].RemoveDuplicates();
      }
      else if ( mpigetrank()%ipow(2, level+1) == 0 && mpigetrank() + ipow(2, level) >= mpigetsize()) {
	continue ;
      } 
      else if ( mpigetrank()%ipow(2, level) == 0) {
	int toproc = mpigetrank()-ipow(2,level);
	world.send(toproc, mpigetsize()*level+mpigetrank(), uniqueDEH[0]);
      }
    }
    
    
    mpi::broadcast(world, uniqueDEH[0], 0);
#endif
    vector<Determinant>& newDets = *uniqueDEH[0].Det;
    ci.conservativeResize(ci.rows()+newDets.size(), 1);

    vector<Determinant>::iterator vec_it = Dets.begin();
    int ciindex = 0, initialSize = Dets.size();
    double EPTguess = 0.0;
    for (vector<Determinant>::iterator it=newDets.begin(); it!=newDets.end(); ) {
      if (schd.excitation != 1000 ) {
	if (it->ExcitationDistance(Dets[0]) > schd.excitation) continue;
      }
      Dets.push_back(*it);
      ci(initialSize+ciindex,0) = uniqueDEH[0].Num->at(ciindex)/(E0 - uniqueDEH[0].Energy->at(ciindex));  
      ciindex++; it++;
    }

  }

  uniqueDEH[0].clear();
  return finalE;
}

void HCIbasics::UpdateRDMPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					      twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec, MatrixXd& twoRDM) {
  // Similar to above, but instead of computing PT energy, update 2RDM
  // AAH, 30 Jan 2017

  boost::mpi::communicator world;
  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  char psiArray[norbs]; vector<int> psiClosed(nelec,0), psiOpen(norbs-nelec,0);
  double energyEN = 0.0;
  int num_thrds = omp_get_max_threads();
  
  
  std::vector<StitchDEH> uniqueDEH(num_thrds);
  double totalPT = 0.0;
#pragma omp parallel 
  {
    uniqueDEH[omp_get_thread_num()].extra_info = true;

    for (int i=0; i<Dets.size(); i++) {

      if (i%(omp_get_num_threads()) != omp_get_thread_num()) {continue;}
      HCIbasics::getPTDeterminantsKeepRefDets(Dets[i], i, abs(schd.epsilon2/ci(i,0)), ci(i,0),
				   I1, I2, I2HB, irrep, coreE, E0,
				   *uniqueDEH[omp_get_thread_num()].Det,
				   *uniqueDEH[omp_get_thread_num()].Num,
				   *uniqueDEH[omp_get_thread_num()].Energy,
				   *uniqueDEH[omp_get_thread_num()].var_indices,
				   *uniqueDEH[omp_get_thread_num()].orbDifference,
				   schd, nelec);

      if (i%100000 == 0 && omp_get_thread_num()==0 && mpigetrank() == 0) cout <<"# "<<i<<endl;

    } // for i
    
    if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#Before sort "<<getTime()-startofCalc<<endl;

    uniqueDEH[omp_get_thread_num()].QuickSortAndRemoveDuplicates();
    uniqueDEH[omp_get_thread_num()].RemoveDetsPresentIn(SortedDets);

    if (mpigetrank() == 0 && omp_get_thread_num() == 0) cout << "#Unique determinants "<<getTime()-startofCalc<<"  "<<endl;
    

    // Merge all PT dets so RDM can be computed by thread 0
#pragma omp barrier 
    if (omp_get_thread_num() == 0) {
      for (int thrd=1; thrd<num_thrds; thrd++) {
	uniqueDEH[0].merge(uniqueDEH[thrd]);
	uniqueDEH[thrd].clear();
	uniqueDEH[0].RemoveDuplicates();
      }
    }
  }

  if (mpigetrank() == 0 ) cout << "#Before mpi split "<<getTime()-startofCalc<<"  "<<uniqueDEH[0].Det->size()<<endl;
  uniqueDEH.resize(1);
  
  
  vector<Determinant>& uniqueDets = *uniqueDEH[0].Det;
  vector<double>& uniqueNumerator = *uniqueDEH[0].Num;
  vector<double>& uniqueEnergy = *uniqueDEH[0].Energy;
  vector<vector<int>>& uniqueVarIndices = *uniqueDEH[0].var_indices;
  vector<vector<int>>& uniqueOrbDiff = *uniqueDEH[0].orbDifference;

// At this point, uniqueDets contains all the dets in the PT space, uniqueNumerator contains all the sum_i^eps2 H_ki c_i values,
// uniqueEnergy contains all their diagonal H elements, and uniqueVarIndices contains all the indices of variational dets
// Finally, uniqueOrbDiff contains the orbitals that are excited between D_i and D_k
// connected to each PT det


  size_t numDets=0, numLocalDets=uniqueDets.size();
  mpi::all_reduce(world, numLocalDets, numDets, std::plus<size_t>());
  if (mpigetrank() == 0) cout <<"#num dets "<<numDets<<endl;
//#pragma omp parallel
  {
    for (size_t k=0; k<uniqueDets.size();k++) {
      //if (k%(omp_get_num_threads()) != omp_get_thread_num()) continue;
      for (size_t i=0; i<uniqueVarIndices[k].size(); i++){
        // Get closed for determinant D_i = Dets[uniqueVarIndices[k][i]]
        // Determine whether single or double excitation 
        int d0=uniqueOrbDiff[k][i]%norbs, c0=(uniqueOrbDiff[k][i]/norbs)%norbs; // These orbitals correspond to an excitation from D_k to D_i
        if (uniqueOrbDiff[k][i]/norbs/norbs == 0) { // single excitation
          vector<int> closed(nelec, 0);
          vector<int> open(norbs-nelec,0);
          Dets[uniqueVarIndices[k][i]].getOpenClosed(open, closed);
	  for (int n1=0;n1<nelec; n1++) {
	    double sgn = 1.0;
	    int a=max(closed[n1],c0), b=min(closed[n1],c0), I=max(closed[n1],d0), J=min(closed[n1],d0); 
	    if (closed[n1] == d0) continue;
	    Dets[uniqueVarIndices[k][i]].parity(min(d0,c0), max(d0,c0),sgn);
	    if (!( (closed[n1] > c0 && closed[n1] > d0) || (closed[n1] < c0 && closed[n1] < d0))) sgn *=-1.;
          //if (a<b or I<J){cout << "Error!" <<endl;}
	    twoRDM(a*(a+1)/2+b, I*(I+1)/2+J) += 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]);
	    twoRDM(I*(I+1)/2+J, a*(a+1)/2+b) += 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]);
	  } // for n1
        }  // single
        else { // double excitation
	  int d1=(uniqueOrbDiff[k][i]/norbs/norbs)%norbs, c1=(uniqueOrbDiff[k][i]/norbs/norbs/norbs)%norbs ;
	  double sgn = 1.0;
	  Dets[uniqueVarIndices[k][i]].parity(d1,d0,c1,c0,sgn);
          int P = max(c1,c0), Q = min(c1,c0), R = max(d1,d0), S = min(d1,d0);
          if (P != c0)  sgn *= -1;
          if (Q != d0)  sgn *= -1;
	  twoRDM(P*(P+1)/2+Q, R*(R+1)/2+S) += 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]);
	  twoRDM(R*(R+1)/2+S, P*(P+1)/2+Q) += 0.5*sgn*uniqueNumerator[k]*ci(uniqueVarIndices[k][i],0)/(E0-uniqueEnergy[k]);
        }// if
      } // i in variational connections to PT det k
    } // k in PT dets
  }

  //mpi::all_reduce(world, totalPT, finalE, std::plus<double>());
  
  return;

}


void HCIbasics::MakeHfromHelpers(std::map<HalfDet, std::vector<int> >& BetaN,
		      std::map<HalfDet, std::vector<int> >& AlphaNm1,
		      std::vector<Determinant>& Dets,
		      int StartIndex,
		      std::vector<std::vector<int> >&connections,
		      std::vector<std::vector<double> >& Helements,
		      int Norbs,
		      oneInt& I1,
		      twoInt& I2,
		      double& coreE,
		      std::vector<std::vector<size_t> >& orbDifference,
		      bool DoRDM) {

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
      //double hij = Energy(&detChar[norbs*k], Norbs, I1, I2, coreE);
      double hij = Dets[k].Energy(I1, I2, coreE);
      Helements[k].push_back(hij);
      if (DoRDM) orbDifference[k].push_back(0);
    }
  }

  std::map<HalfDet, std::vector<int> >::iterator ita = BetaN.begin();
  int index = 0;
  pout <<"# "<< Dets.size()<<"  "<<BetaN.size()<<"  "<<AlphaNm1.size()<<endl;
  pout << "#";
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
    if (index%1000000 == 0 && index!= 0) {pout <<". ";}
  }
  pout << format("BetaN    %49.2f\n#")
      % (getTime()-startofCalc);

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
    if (index%1000000 == 0 && index!= 0) {pout <<". ";}
  }

  pout << format("AlphaN-1 %49.2f\n")
      % (getTime()-startofCalc);


    
}
  
void HCIbasics::PopulateHelperLists(std::map<HalfDet, std::vector<int> >& BetaN,
				    std::map<HalfDet, std::vector<int> >& AlphaNm1,
				    std::vector<Determinant>& Dets,
				    int StartIndex) {
  pout << format("#Making Helpers %43.2f\n")
      % (getTime()-startofCalc);
  for (int i=StartIndex; i<Dets.size(); i++) {
    HalfDet da = Dets[i].getAlpha(), db = Dets[i].getBeta();

    BetaN[db].push_back(i);

    int norbs = 64*DetLen;
    std::vector<int> closeda(norbs/2);//, closedb(norbs);
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
//it then does a HCI varitional calculation and the resulting
//ci and dets are returned here
//at input usually the Dets will just have a HF or some such determinant
//and ci will be just 1.0
vector<double> HCIbasics::DoVariational(vector<MatrixXd>& ci, vector<Determinant>& Dets, schedule& schd,
					  twoInt& I2, twoIntHeatBath& I2HB, vector<int>& irrep, oneInt& I1, double& coreE
					  , int nelec, bool DoRDM) {

  int nroots = ci.size();
  std::map<HalfDet, std::vector<int> > BetaN, AlphaNm1;
  PopulateHelperLists(BetaN, AlphaNm1, Dets, 0);

#ifndef SERIAL
  boost::mpi::communicator world;
#endif
  int num_thrds = omp_get_max_threads();

  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());

  size_t norbs = 2.*I2.Direct.rows();
  int Norbs = norbs;
  //std::vector<char> detChar(norbs); Dets[0].getRepArray(&detChar[0]);
  vector<double> E0(nroots,Dets[0].Energy(I1, I2, coreE));

  pout << "#HF = "<<E0[0]<<std::endl;

  //this is essentially the hamiltonian, we have stored it in a sparse format
  std::vector<std::vector<int> > connections; connections.resize(Dets.size());
  std::vector<std::vector<double> > Helements;Helements.resize(Dets.size());
  std::vector<std::vector<size_t> > orbDifference;orbDifference.resize(Dets.size());
  MakeHfromHelpers(BetaN, AlphaNm1, Dets, 0, connections, Helements,
		   norbs, I1, I2, coreE, orbDifference, DoRDM);
  //if (mpigetrank() != 0) {connections.resize(0); Helements.resize(0);orbDifference.resize(0);}

  //keep the diagonal energies of determinants so we only have to generated
  //this for the new determinants in each iteration and not all determinants
  MatrixXd diagOld(Dets.size(),1); 
  for (int i=0; i<Dets.size(); i++)
    diagOld(i,0) = Dets[i].Energy(I1, I2, coreE);
  int prevSize = 0;

  int iterstart = 0;
  if (schd.restart || schd.fullrestart) {
    bool converged;
    readVariationalResult(iterstart, ci, Dets, SortedDets, diagOld, connections, orbDifference, Helements, E0, converged, schd, BetaN, AlphaNm1);
    if (schd.fullrestart)
      iterstart = 0;
    pout << format("# %4i  %10.2e  %10.2e   %14.8f  %10.2f\n") 
      %(iterstart) % schd.epsilon1[iterstart] % Dets.size() % E0[0] % (getTime()-startofCalc);
    if (!schd.fullrestart)
      iterstart++;
    if (schd.onlyperturbative)
      return E0;

    if (converged && iterstart >= schd.epsilon1.size()) {
      pout << "# restarting from a converged calculation, moving to perturbative part.!!"<<endl;
      return E0;
    }
  }


  //do the variational bit
  for (int iter=iterstart; iter<schd.epsilon1.size(); iter++) {
    double epsilon1 = schd.epsilon1[iter];
    //std::vector<vector<Determinant> > newDets(num_thrds); //also include the connection magnitude so we can calculate the pt
    std::vector<StitchDEH> uniqueDEH(num_thrds);

    pout << format("#-------------Iter=%4i---------------") % iter<<endl;
    MatrixXd cMax(ci[0].rows(),1); cMax = 1.*ci[0];
    for (int i=1; i<ci.size(); i++)
      for (int j=0; j<ci[0].rows(); j++)
	cMax(j,0) = max(fabs(cMax(j,0)), fabs(ci[i](j,0)) );

#pragma omp parallel 
    {
      for (int i=0; i<SortedDets.size(); i++) {
	if (i%(mpigetsize()*omp_get_num_threads()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) continue;
	HCIbasics::getDeterminants(Dets[i], abs(epsilon1/cMax(i,0)), cMax(i,0), 0.0, 
				   I1, I2, I2HB, irrep, coreE, E0[0], 
				   *uniqueDEH[omp_get_thread_num()].Det, 
				   *uniqueDEH[omp_get_thread_num()].Num, 
				   *uniqueDEH[omp_get_thread_num()].Energy, 
				   schd,0, nelec, false);
      }

      
      uniqueDEH[omp_get_thread_num()].QuickSortAndRemoveDuplicates();
      uniqueDEH[omp_get_thread_num()].RemoveDetsPresentIn(SortedDets);

#pragma omp barrier 
      if (omp_get_thread_num() == 0) {
	for (int thrd=1; thrd<num_thrds; thrd++) {
	  uniqueDEH[0].merge(uniqueDEH[thrd]);
	  uniqueDEH[thrd].clear();
	  uniqueDEH[0].RemoveDuplicates();
	}
      }
    }

    uniqueDEH.resize(1);

#ifndef SERIAL
    for (int level = 0; level <ceil(log2(mpigetsize())); level++) {
      
      if (mpigetrank()%ipow(2, level+1) == 0 && mpigetrank() + ipow(2, level) < mpigetsize()) {
	StitchDEH recvDEH;	
	int getproc = mpigetrank()+ipow(2,level);
	world.recv(getproc, mpigetsize()*level+getproc, recvDEH);
	uniqueDEH[0].merge(recvDEH);
	uniqueDEH[0].RemoveDuplicates();
      }
      else if ( mpigetrank()%ipow(2, level+1) == 0 && mpigetrank() + ipow(2, level) >= mpigetsize()) {
	continue ;
      } 
      else if ( mpigetrank()%ipow(2, level) == 0) {
	int toproc = mpigetrank()-ipow(2,level);
	world.send(toproc, mpigetsize()*level+mpigetrank(), uniqueDEH[0]);
      }
    }


    mpi::broadcast(world, uniqueDEH[0], 0);
#endif

    vector<Determinant>& newDets = *uniqueDEH[0].Det;
    //cout << newDets.size()<<"  "<<Dets.size()<<endl;
    vector<MatrixXd> X0(ci.size(), MatrixXd(Dets.size()+newDets.size(), 1)); 
    for (int i=0; i<ci.size(); i++) {
      X0[i].setZero(Dets.size()+newDets.size(),1); 
      X0[i].block(0,0,ci[i].rows(),1) = 1.*ci[i]; 
      
    }

    vector<Determinant>::iterator vec_it = SortedDets.begin();
    int ciindex = 0;
    double EPTguess = 0.0;
    for (vector<Determinant>::iterator it=newDets.begin(); it!=newDets.end(); ) {
      if (schd.excitation != 1000 ) {
	if (it->ExcitationDistance(Dets[0]) > schd.excitation) continue;
      }
      Dets.push_back(*it);

      if (iter != 0) {
	for (int i=0; i<ci.size(); i++) {
	  X0[i](ci[i].rows()+ciindex,0) = uniqueDEH[0].Num->at(ciindex)/(E0[i] - uniqueDEH[0].Energy->at(ciindex));  
	  EPTguess += pow(uniqueDEH[0].Num->at(ciindex), 2)/(E0[i] - uniqueDEH[0].Energy->at(ciindex));  
	}
      }
      ciindex++;vec_it++; it++;
    }

    if (iter != 0) pout << str(boost::format("#Initial guess(PT) : %18.10g  \n") %(E0[0]+EPTguess) );
    uniqueDEH.resize(0);
    //now diagonalize the hamiltonian

    MatrixXd diag(Dets.size(), 1); diag.setZero(diag.size(),1);
    if (mpigetrank() == 0) diag.block(0,0,ci[0].rows(),1)= 1.*diagOld;


    double estimatedCorrection = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+:estimatedCorrection)
    for (size_t k=SortedDets.size(); k<Dets.size() ; k++) {
      if (k % mpigetsize() != mpigetrank() ) continue;
      diag(k,0) = Dets[k].Energy(I1, I2, coreE);
      if (k%1000000 == 0 && k!=0) cout <<"#"<< k<<"Hdiag out of "<<Dets.size()<<endl;     
    }

#ifndef SERIAL
    MPI_Allreduce(MPI_IN_PLACE, &diag(0,0), diag.rows(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    connections.resize(Dets.size());
    Helements.resize(Dets.size());
    orbDifference.resize(Dets.size());


    
    PopulateHelperLists(BetaN, AlphaNm1, Dets, ci[0].size());
    MakeHfromHelpers(BetaN, AlphaNm1, Dets, SortedDets.size(), connections, Helements,
		     norbs, I1, I2, coreE, orbDifference, DoRDM);
     
    for (size_t i=SortedDets.size(); i<Dets.size(); i++)
      SortedDets.push_back(Dets[i]);
    std::sort(SortedDets.begin(), SortedDets.end());
 

    
    double prevE0 = E0[0];
    //Hmult H(&detChar[0], norbs, I1, I2, coreE);
    Hmult2 H(connections, Helements);

    E0 = davidson(H, X0, diag, schd.nroots+10, schd.davidsonTol, false);

    //E0[0] = davidson(H, X0[0], diag, schd.nroots+10, schd.davidsonTol, false);
    for (int i=0; i<E0.size(); i++) {
      ci[i].resize(Dets.size(),1); ci[i] = 1.0*X0[i];
      X0[i].resize(0,0);
    }

    diagOld.resize(Dets.size(),1); diagOld = 1.0*diag;

    if (abs(E0[0]-prevE0) < schd.dE || iter == schd.epsilon1.size()-1)  {

      writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, orbDifference, Helements, E0, true, schd, BetaN, AlphaNm1);
      if (DoRDM) {	
	Helements.resize(0); BetaN.clear(); AlphaNm1.clear();
	for (int i=0; i<schd.nroots; i++) {
          MatrixXd twoRDM(norbs*(norbs+1)/2, norbs*(norbs+1)/2);
	  EvaluateAndStoreRDM(connections, Dets, ci[i], orbDifference, nelec, schd, i, twoRDM);
          cout << "Variational RDM:" << endl;
          ComputeEnergyFromRDM(norbs, nelec, I1, I2, coreE, twoRDM);
          UpdateRDMPerturbativeDeterministic(Dets, ci[i], E0[0], I1, I2, I2HB, irrep, schd, coreE, nelec, twoRDM);
          cout << "Var+PT RDM:" << endl;
          ComputeEnergyFromRDM(norbs, nelec, I1, I2, coreE, twoRDM);
          printRDM(Dets[0].norbs, schd, i, twoRDM);
        } // for i
      }

      break;
    }
    else {
      if (schd.io) writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, orbDifference, Helements, E0, false, schd, BetaN, AlphaNm1);
    }

    pout << format("###########################################      %10.2f ") %(getTime()-startofCalc)<<endl;
  }
  return E0;

}


void HCIbasics::writeVariationalResult(int iter, vector<MatrixXd>& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
				       MatrixXd& diag, vector<vector<int> >& connections, vector<vector<size_t> >&orbdifference, 
				       vector<vector<double> >& Helements, 
				       vector<double>& E0, bool converged, schedule& schd,   
				       std::map<HalfDet, std::vector<int> >& BetaN, 
				       std::map<HalfDet, std::vector<int> >& AlphaNm1) {
  
#ifndef SERIAL
  boost::mpi::communicator world;
#endif

/*
  cout << "Variational wavefunction" << endl;
  for (int root=0; root<schd.nroots; root++) {
    pout << "### IMPORTANT DETERMINANTS FOR STATE: "<<root<<endl;
    MatrixXd prevci = 1.*ci[root];
    for (int i=0; i<5; i++) {
      compAbs comp;
      int m = distance(&prevci(0,0), max_element(&prevci(0,0), &prevci(0,0)+prevci.rows(), comp));
      pout <<"#"<< i<<"  "<<prevci(m,0)<<"  "<<Dets[m]<<endl;
      prevci(m,0) = 0.0;
    }
  }
*/

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
    save << ci;
    save << E0;
    save << converged;
    save << connections<<orbdifference<<Helements;
    save << BetaN<< AlphaNm1;
    ofs.close();

    pout << format("#End   writing variational wf %29.2f\n")
      % (getTime()-startofCalc);
}


void HCIbasics::readVariationalResult(int& iter, vector<MatrixXd>& ci, vector<Determinant>& Dets, vector<Determinant>& SortedDets,
				      MatrixXd& diag, vector<vector<int> >& connections, vector<vector<size_t> >& orbdifference, 
				      vector<vector<double> >& Helements, 
					vector<double>& E0, bool& converged, schedule& schd,
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
    ci.resize(1, MatrixXd(diaglen,1)); diag.resize(diaglen,1);
    for (int i=0; i<diag.rows(); i++)
      load >> diag(i,0);

    load >> ci;
    load >> E0;
    if (schd.onlyperturbative) {ifs.close();return;}
    load >> converged;

    load >> connections >> orbdifference >> Helements;
    load >> BetaN>> AlphaNm1;
    ifs.close();

    pout << format("#End   reading variational wf %29.2f\n")
      % (getTime()-startofCalc);
}



//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void HCIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<double>& numerator, std::vector<double>& energy, schedule& schd, int Nmc, int nelec, bool mpispecific) {

  int norbs = d.norbs;
  //char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  //d.getRepArray(detArray);

  double Energyd = Energy(closed, nclosed, int1, int2, coreE);
  bool parallelRegion = (mpigetsize() != 1 && mpispecific) ? true : false;

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    //if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;

    double integral = Hij_1Excite(closed[i],open[a],int1,int2, &closed[0], nclosed);
    

    if (fabs(integral) > epsilon ) {
      dets.push_back(d); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      if (parallelRegion) {
	if (di.getLexicalOrder()%(mpigetsize()) != mpigetrank()) {
	  dets.pop_back();
	  continue;
	}
      }


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
	//if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
	
	if (!(d.getocc(a) || d.getocc(b))) {
	  dets.push_back(d);
	  Determinant& di = *dets.rbegin();
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    

	  if (parallelRegion) {
	    if (di.getLexicalOrder()%(mpigetsize()) != mpigetrank()) {
	      dets.pop_back();
	      continue;
	    }
	  }

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

void HCIbasics::getPTDeterminantsKeepRefDets(Determinant det, int det_ind, double epsilon, double ci, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<double>& numerator, std::vector<double>& energy, std::vector<std::vector<int> >& var_indices, std::vector<std::vector<int> >& orbDifference, schedule& schd, int nelec) {
  // Similar to above subroutine, but also keeps track of the reference dets each connected det came from
  // AAH, 30 Jan 2017

  int norbs = det.norbs;
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  det.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  int orbDiff;

  double Energyd = Energy(closed, nclosed, int1, int2, coreE);
  bool parallelRegion = mpigetsize() == 1 ? false : true;
  std::vector<int> var_indices_vec;
  std::vector<int> orbDiff_vec;

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    //if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;

    double integral = Hij_1Excite(closed[i],open[a],int1,int2, &closed[0], nclosed);
    

    if (fabs(integral) > epsilon ) {
      dets.push_back(det); Determinant& di = *dets.rbegin();
      di.setocc(open[a], true); di.setocc(closed[i],false);

      if (parallelRegion) {
	if (di.getLexicalOrder()%(mpigetsize()) != mpigetrank()) {
	  dets.pop_back();
	  continue;
	}
      }

      double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, open[a], Energyd);

      numerator.push_back(integral*ci);
      energy.push_back(E);

      std::vector<int> var_indices_vec;
      var_indices_vec.push_back(det_ind);
      var_indices.push_back(var_indices_vec);

      orbDiff = open[a]*norbs+closed[i]; // a = creation, i = annihilation
      std::vector<int> orbDiff_vec;
      orbDiff_vec.push_back(orbDiff);
      orbDifference.push_back(orbDiff_vec);

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
	
	if (!(det.getocc(a) || det.getocc(b))) {
	  dets.push_back(det);
	  Determinant& di = *dets.rbegin();
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    

	  if (parallelRegion) {
	    if (di.getLexicalOrder()%(mpigetsize()) != mpigetrank()) {
	      dets.pop_back();
	      continue;
	    }
	  }

	  double sgn = 1.0;
	  di.parity(a, b, closed[i], closed[j], sgn);
	  numerator.push_back(it->first*sgn*ci);
	  
	  double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);	    
	  energy.push_back(E);

          std::vector<int> var_indices_vec;
          var_indices_vec.push_back(det_ind);
          var_indices.push_back(var_indices_vec);

          orbDiff = a*norbs*norbs*norbs+closed[i]*norbs*norbs+b*norbs+closed[j];  //i>j and a>b??
          std::vector<int> orbDiff_vec;
          orbDiff_vec.push_back(orbDiff);
          orbDifference.push_back(orbDiff_vec);
	  
	} // if a and b unoccupied
      } //for ???
    } // iterate over heatbath integrals
  } // i and j
  return;
}
/*
//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void HCIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<double>& numerator, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

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

      if (di.getLexicalOrder()/(mpigetsize()*omp_get_num_threads()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) {
	dets.pop_back();
	continue;
      }

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

	  if (di.getLexicalOrder()/(mpigetsize()*omp_get_num_threads()) != mpigetrank()*omp_get_num_threads()+omp_get_thread_num()) {
	    dets.pop_back();
	    continue;
	  }
	  
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
*/

//this function is complicated because I wanted to make it general enough that deterministicperturbative and stochasticperturbative could use the same function
//in stochastic perturbative each determinant in Psi1 can come from the first replica of Psi0 or the second replica of Psi0. that is why you have a pair of doubles associated with each det in Psi1 and we pass ci1 and ci2 which are the coefficients of d in replica 1 and replica2 of Psi0.
void HCIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  //char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  //d.getRepArray(detArray);

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    //if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
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
	//if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital

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
void HCIbasics::getDeterminants(Determinant& d, double epsilon, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<double>& numerator1, vector<double>& numerator2, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  //char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  //d.getRepArray(detArray);

  double Energyd = Energy(closed, nclosed, int1, int2, coreE);
  

  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    //if (open[a]/2 > schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital
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
	//if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital

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
void HCIbasics::getDeterminants2Epsilon(Determinant& d, double epsilon, double epsilonLarge, double ci1, double ci2, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::vector<Determinant>& dets, std::vector<double>& numerator1A, vector<double>& numerator2A, vector<bool>& present, std::vector<double>& energy, schedule& schd, int Nmc, int nelec) {

  int norbs = d.norbs;
  //char detArray[norbs], diArray[norbs];
  vector<int> closed(nelec,0); 
  vector<int> open(norbs-nelec,0);
  d.getOpenClosed(open, closed);
  int nclosed = nelec;
  int nopen = norbs-nclosed;
  //d.getRepArray(detArray);

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

      if (fabs(integral) >epsilonLarge) 
	present.push_back(true);
      else 
	present.push_back(false);

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
	//if (a/2 > schd.nvirt+nclosed/2 || b/2 >schd.nvirt+nclosed/2) continue; //dont occupy above a certain orbital

	if (!(d.getocc(a) || d.getocc(b))) {
	  dets.push_back(d);
	  Determinant& di = *dets.rbegin();
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    

	  double sgn = 1.0;
	  di.parity(a, b, closed[i], closed[j], sgn);

	  numerator1A.push_back(it->first*sgn*ci1);
	  numerator2A.push_back(it->first*it->first*ci1*(ci1*Nmc/(Nmc-1)-ci2));


	  if (fabs(it->first) >epsilonLarge) 
	    present.push_back(true);
	  else 
	    present.push_back(false);

	  double E = EnergyAfterExcitation(closed, nclosed, int1, int2, coreE, i, a, j, b, Energyd);	    
	  energy.push_back(E);

	}
      }
    }
  }
  return;
}


