/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
  Copyright (c) 2017, Sandeep Sharma
  
  This file is part of DICE.
  
  This program is free software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation, 
  either version 3 of the License, or (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with this program. 
  If not, see <http://www.gnu.org/licenses/>.
*/
#include "Determinants.h"
#include "SHCIbasics.h"
#include "SHCIgetdeterminants.h"
#include "SHCIrdm.h"
#include "SHCISortMpiUtils.h"
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


namespace SHCISortMpiUtils {


  int binarySearch(int* arr, int l, int r, int x)
  {
    if (r >= l)
      {
        int mid = l + (r - l)/2;
 
        // If the element is present at the middle itself
        if (arr[mid] == x)  return mid;
 
        // If element is smaller than mid, then it can only be present
        // in left subarray
        if (arr[mid] > x) return binarySearch(arr, l, mid-1, x);
 
        // Else the element can only be present in right subarray
        return binarySearch(arr, mid+1, r, x);
      }
 
    // We reach here when element is not present in array
    return -1;
  }

  void RemoveDuplicates(std::vector<Determinant>& Det,
			std::vector<double>& Num1, std::vector<double>& Num2,
			std::vector<double>& Energy, std::vector<char>& present) {

    size_t uniqueSize = 0;
    for (size_t i=1; i <Det.size(); i++) {
      if (!(Det[i] == Det[i-1])) {
	uniqueSize++;
	Det.at(uniqueSize) = Det[i];
	Num1.at(uniqueSize) = Num1[i];
	Num2.at(uniqueSize) = Num2[i];
	Energy.at(uniqueSize) = Energy[i];
	present.at(uniqueSize) = present[i];
      }
      else {
	Num1.at(uniqueSize) += Num1[i];
	Num2.at(uniqueSize) += Num1[i];
      }
    }
    Det.resize(uniqueSize+1);
    Num1.resize(uniqueSize+1);
    Num2.resize(uniqueSize+1);
    Energy.resize(uniqueSize+1);
    present.resize(uniqueSize+1);
  }

  /*
  void RemoveDetsPresentIn(std::vector<Determinant>& SortedDets, std::vector<Determinant>& Det,
			   std::vector<double>& Num1, std::vector<double>& Num2,
			   std::vector<double>& Energy, std::vector<char>& present) {

    vector<Determinant>::iterator vec_it = SortedDets.begin();
    vector<Determinant>::iterator vec_end = SortedDets.end();

    size_t uniqueSize = 0;
    for (size_t i=0; i<Det.size();) {
      if (Det[i] < *vec_it) {
	Det[uniqueSize] = Det[i];
	Num1[uniqueSize] = Num1[i];
	Num2[uniqueSize] = Num2[i];
	Energy[uniqueSize] = Energy[i];
	present[uniqueSize] = present[i];
	i++; uniqueSize++;
      }
      else if (*vec_it < Det[i] && vec_it != vec_end)
	vec_it ++;
      else if (*vec_it < Det[i] && vec_it == vec_end) {
	Det[uniqueSize] = Det[i];
	Num1[uniqueSize] = Num1[i];
	Num2[uniqueSize] = Num2[i];
	Energy[uniqueSize] = Energy[i];
	present[uniqueSize] = present[i];
	i++; uniqueSize++;
      }
      else {
	i++;
      }
    }
    Det.resize(uniqueSize); Num1.resize(uniqueSize);
    Num2.resize(uniqueSize); Energy.resize(uniqueSize);
    present.resize(uniqueSize);
  }
  */
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

  int partition(Determinant* A, int p,int q, CItype* pNum, double* Energy,
		vector<double>* det_energy, vector<char>* present)
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

	    if (present != NULL)
	      swap(present->at(i), present->at(j));
	    if (det_energy != NULL)
	      swap(det_energy->at(i), det_energy->at(j));
	  }

      }

    swap(A[i],A[p]);
    swap(pNum[i], pNum[p]);
    swap(Energy[i], Energy[p]);

    if (present != NULL)
      swap(present->at(i), present->at(p));
    if (det_energy != NULL)
      swap(det_energy->at(i), det_energy->at(p));
    return i;
  }

  int partitionAll(Determinant* A, int p,int q, CItype* pNum, double* Energy,
		   vector<int>* var_indices, vector<size_t>* orbDifference,
		   vector<double>* det_energy, vector<bool>* present)
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

  void quickSort(Determinant* A, int p,int q, CItype* pNum, double* Energy,
		 vector<double>* det_energy, vector<char>* present)
  {
    int r;
    if(p<q)
      {
	r=partition(A,p,q, pNum, Energy, det_energy, present);
	quickSort(A,p,r, pNum, Energy, det_energy, present);
	quickSort(A,r+1,q, pNum, Energy, det_energy, present);
      }
  }

  void quickSortAll(Determinant* A, int p,int q, CItype* pNum, double* Energy,
		    vector<int>* var_indices, vector<size_t>* orbDifference,
		    vector<double>* det_energy, vector<bool>* present)
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

  void merge(int *a, long low, long high, long mid, int* x, int* c, int* cx)
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

  void mergesort(int *a, long low, long high, int* x, int* c, int* cx)
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



  StitchDEH::StitchDEH() {
    Det = std::shared_ptr<vector<Determinant> > (new vector<Determinant>() );
    Num = std::shared_ptr<vector<CItype> > (new vector<CItype>() );
    Num2 = std::shared_ptr<vector<CItype> > (new vector<CItype>() );
    present = std::shared_ptr<vector<char> > (new vector<char>() );
    Energy = std::shared_ptr<vector<double> > (new vector<double>() );
    extra_info = false;
    var_indices = std::shared_ptr<vector<vector<int> > > (new vector<vector<int> >() );
    orbDifference = std::shared_ptr<vector<vector<size_t> > > (new vector<vector<size_t> >() );
    var_indices_beforeMerge = std::shared_ptr<vector<int > > (new vector<int >() );
    orbDifference_beforeMerge = std::shared_ptr<vector<size_t > > (new vector<size_t >() );
  }

  StitchDEH::StitchDEH(std::shared_ptr<vector<Determinant> >pD,
		       std::shared_ptr<vector<CItype> >pNum,
		       std::shared_ptr<vector<CItype> >pNum2,
		       std::shared_ptr<vector<char> >ppresent,
		       std::shared_ptr<vector<double> >pE,
		       std::shared_ptr<vector<vector<int> > >pvar,
		       std::shared_ptr<vector<vector<size_t> > >porb,
		       std::shared_ptr<vector<int > >pvar_beforeMerge,
		       std::shared_ptr<vector<size_t > >porb_beforeMerge)
    : Det(pD), Num(pNum), Num2(pNum2), present(ppresent), Energy(pE), var_indices(pvar), orbDifference(porb), var_indices_beforeMerge(pvar_beforeMerge), orbDifference_beforeMerge(porb_beforeMerge)
  {
    extra_info=true;
  };

  StitchDEH::StitchDEH(std::shared_ptr<vector<Determinant> >pD,
		       std::shared_ptr<vector<CItype> >pNum,
		       std::shared_ptr<vector<CItype> >pNum2,
		       std::shared_ptr<vector<char> >ppresent,
		       std::shared_ptr<vector<double> >pE)
    : Det(pD), Num(pNum), Num2(pNum2), present(ppresent), Energy(pE)
  {
    extra_info=false;
  };

  void StitchDEH::QuickSortAndRemoveDuplicates() {
    if (Num2->size() > 0) {
      cout << "Use mergesort instead"<<endl;
      exit(0);
    }

    if (extra_info) {
      quickSortAll(&(Det->operator[](0)), 0, Det->size(), &(Num->operator[](0)), &(Energy->operator[](0)), &(var_indices->operator[](0)), &(orbDifference->operator[](0)));
    } else {
      quickSort(&(Det->operator[](0)), 0, Det->size(), &(Num->operator[](0)), &(Energy->operator[](0)));
    }

    //if (Det->size() == 1) return;
    if (Det->size() <= 1) return;

    std::vector<Determinant>& Detcopy = *Det;
    std::vector<CItype>& Numcopy = *Num;
    std::vector<double>& Ecopy = *Energy;
    //if (extra_info) {
    std::vector<std::vector<int> >& Vcopy = *var_indices;
    std::vector<std::vector<size_t> >& Ocopy = *orbDifference;
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

  void StitchDEH::MergeSort() {
    std::vector<Determinant> Detcopy = *Det;

    std::vector<long> detIndex(Detcopy.size(),0);
    std::vector<long> detIndexcopy(Detcopy.size(),0);

    for (size_t i=0; i<Detcopy.size(); i++)
      detIndex[i] = i;
    mergesort(&Detcopy[0], 0, Detcopy.size()-1, &detIndex[0], &( Det->operator[](0)), &detIndexcopy[0]);
    detIndexcopy.clear();

    bool varIndicesEmtpy = var_indices->size() == 0 ? true : false;

    if (Det->size() <= 1) return;
    reorder(*Num, detIndex);
    if (Num2->size() != 0)
      reorder(*Num2, detIndex);
    if (present->size() != 0)
      reorder(*present, detIndex);
    reorder(*Energy, detIndex);
    if (extra_info) {
      if (varIndicesEmtpy) {
	reorder(*var_indices_beforeMerge, detIndex);
	reorder(*orbDifference_beforeMerge, detIndex);
      }
      else {
	vector<vector<int> > Vcopy = *var_indices;
	for (int i=0; i<Vcopy.size(); i++) {
	  var_indices->at(i).clear();
	  var_indices->at(i).insert(var_indices->at(i).end(), Vcopy[detIndex[i]].begin(), Vcopy[detIndex[i]].end());
	}
	Vcopy.clear();
	vector<vector<size_t> > Ocopy = *orbDifference;
	for (int i=0; i<Ocopy.size(); i++) {
	  orbDifference->at(i).clear();
	  orbDifference->at(i).insert(orbDifference->at(i).end(), Ocopy[detIndex[i]].begin(), Ocopy[detIndex[i]].end());
	}
	Ocopy.clear();
      }

    }
    detIndex.clear();

    std::vector<std::vector<int> >& Vcopy = *var_indices;
    std::vector<std::vector<size_t> >& Ocopy = *orbDifference;

    size_t uniqueSize = 0;

    if (extra_info) {

      if (varIndicesEmtpy) {
	var_indices->push_back(std::vector<int>(1,var_indices_beforeMerge->at(0)));
	orbDifference->push_back(std::vector<size_t>(1,orbDifference_beforeMerge->at(0)));
      }
    }

  }

  void StitchDEH::MergeSortAndRemoveDuplicates() {
    std::vector<Determinant> Detcopy = *Det;

    std::vector<long> detIndex(Detcopy.size(),0);
    std::vector<long> detIndexcopy(Detcopy.size(),0);

    for (size_t i=0; i<Detcopy.size(); i++)
      detIndex[i] = i;
    mergesort(&Detcopy[0], 0, Detcopy.size()-1, &detIndex[0], &( Det->operator[](0)), &detIndexcopy[0]);
    detIndexcopy.clear();

    bool varIndicesEmtpy = var_indices->size() == 0 ? true : false;

    if (Det->size() <= 1) return;
    reorder(*Num, detIndex);
    if (Num2->size() != 0)
      reorder(*Num2, detIndex);
    if (present->size() != 0)
      reorder(*present, detIndex);
    reorder(*Energy, detIndex);
    if (extra_info) {
      if (varIndicesEmtpy) {
	reorder(*var_indices_beforeMerge, detIndex);
	reorder(*orbDifference_beforeMerge, detIndex);
      }
      else {
	vector<vector<int> > Vcopy = *var_indices;
	for (int i=0; i<Vcopy.size(); i++) {
	  var_indices->at(i).clear();
	  var_indices->at(i).insert(var_indices->at(i).end(), Vcopy[detIndex[i]].begin(), Vcopy[detIndex[i]].end());
	}
	Vcopy.clear();
	vector<vector<size_t> > Ocopy = *orbDifference;
	for (int i=0; i<Ocopy.size(); i++) {
	  orbDifference->at(i).clear();
	  orbDifference->at(i).insert(orbDifference->at(i).end(), Ocopy[detIndex[i]].begin(), Ocopy[detIndex[i]].end());
	}
	Ocopy.clear();
      }

    }
    detIndex.clear();

    std::vector<std::vector<int> >& Vcopy = *var_indices;
    std::vector<std::vector<size_t> >& Ocopy = *orbDifference;

    size_t uniqueSize = 0;

    if (extra_info) {

      if (varIndicesEmtpy) {
	var_indices->push_back(std::vector<int>(1,var_indices_beforeMerge->at(0)));
	orbDifference->push_back(std::vector<size_t>(1,orbDifference_beforeMerge->at(0)));
      }
    }

    if (varIndicesEmtpy) {
      for (size_t i=1; i <Detcopy.size(); i++) {
	if (!(Detcopy[i] == Detcopy[i-1])) {
	  uniqueSize++;
	  Det->operator[](uniqueSize) = Det->at(i);
	  Num->operator[](uniqueSize) = Num->at(i);
	  if (Num2->size() != 0)
	    Num2->operator[](uniqueSize) = Num2->at(i);
	  if (present->size() != 0)
	    present->operator[](uniqueSize) = present->at(i);
	  Energy->operator[](uniqueSize) = Energy->at(i);
	  if (extra_info) {
	    var_indices->push_back(std::vector<int>(1,var_indices_beforeMerge->at(i)));
	    orbDifference->push_back(std::vector<size_t>(1,orbDifference_beforeMerge->at(i)));
	    //var_indices->operator[](uniqueSize) = Vcopy[detIndex[i]];
	    //orbDifference->operator[](uniqueSize) = Ocopy[detIndex[i]];
	  }
	}
	else {
	  Num->operator[](uniqueSize) += Num->at(i);
	  if (Num2->size() != 0)
	    Num2->operator[](uniqueSize) += Num2->at(i);
	  if (present->size() != 0)
	    present->operator[](uniqueSize) = present->at(i);
	  if (abs(Energy->operator[](uniqueSize) - Energy->at(i)) > 1.e-10 && 
	      abs(Energy->operator[](uniqueSize) - Energy->at(i)) > 1.e-10*abs(Energy->at(i)) ) {
	    cout << uniqueSize<<"  "<<i<<"  "<<endl;
	    cout << Detcopy[i]<<endl;
	    cout << Energy->at(uniqueSize)<<"  "<<Energy->at(i)<<"  "<<Detcopy[i-1]<<endl;
	    cout << "Energy of determinants is not consistent."<<endl;
	    exit(0);
	  }
	  if (extra_info) {
	    var_indices->at(uniqueSize).push_back(var_indices_beforeMerge->at(i));
	    orbDifference->at(uniqueSize).push_back(orbDifference_beforeMerge->at(i));
    	    //(*var_indices)[uniqueSize].push_back(Vcopy[detIndex[i]][k]);
    	    //(*orbDifference)[uniqueSize].push_back(Ocopy[detIndex[i]][k]);
	  }
	}
      }
    }
    else {
      for (size_t i=1; i <Detcopy.size(); i++) {
	if (!(Detcopy[i] == Detcopy[i-1])) {
	  uniqueSize++;
	  Det->operator[](uniqueSize) = Det->at(i);
	  Num->operator[](uniqueSize) = Num->at(i);
	  if (Num2->size() != 0)
	    Num2->operator[](uniqueSize) = Num2->at(i);
	  if (present->size() != 0)
	    present->operator[](uniqueSize) = present->at(i);
	  Energy->operator[](uniqueSize) = Energy->at(i);
	  if (extra_info) {
	    var_indices->operator[](uniqueSize) = Vcopy[i];
	    orbDifference->operator[](uniqueSize) = Ocopy[i];
	  }
	}
	else {
	  Num->operator[](uniqueSize) += Num->at(i);
	  if (Num2->size() != 0)
	    Num2->operator[](uniqueSize) += Num2->at(i);
	  if (present->size() != 0)
	    present->operator[](uniqueSize) = present->at(i);
	  if (extra_info) {
	    for (size_t k=0; k<Vcopy[i].size(); k++) {
	      (*var_indices)[uniqueSize].push_back(Vcopy[i][k]);
	      (*orbDifference)[uniqueSize].push_back(Ocopy[i][k]);
	    }
	  }
	}
      }
    }


    var_indices_beforeMerge->clear();
    orbDifference_beforeMerge->clear();

    Det->resize(uniqueSize+1);
    Num->resize(uniqueSize+1);
    if (Num2->size() != 0)
      Num2->resize(uniqueSize+1);
    if (present->size() != 0)
      present->resize(uniqueSize+1);
    Energy->resize(uniqueSize+1);
    if (extra_info) {
      var_indices->resize(uniqueSize+1);
      orbDifference->resize(uniqueSize+1);
    }
  }


  void StitchDEH::RemoveDetsPresentIn(std::vector<Determinant>& SortedDets) {
    vector<Determinant>::iterator vec_it = SortedDets.begin();
    std::vector<Determinant>& Detcopy = *Det;
    std::vector<CItype>& Numcopy = *Num;
    std::vector<CItype>& Num2copy = *Num2;
    std::vector<char>& presentcopy = *present;
    std::vector<double>& Ecopy = *Energy;
    //if (extra_info) {
    std::vector<std::vector<int> >& Vcopy = *var_indices;
    std::vector<std::vector<size_t> >& Ocopy = *orbDifference;
    //}

    size_t uniqueSize = 0, i=0;
    while (i <Detcopy.size() && vec_it != SortedDets.end()) {
      if (Detcopy[i] < *vec_it) {
	Det->operator[](uniqueSize) = Detcopy[i];
	Num->operator[](uniqueSize) = Numcopy[i];
	if (Num2->size() != 0)
	  Num2->operator[](uniqueSize) = Num2copy[i];
	if (present->size() != 0)
	  present->operator[](uniqueSize) = presentcopy[i];
	Energy->operator[](uniqueSize) = Ecopy[i];
	if (extra_info) {
	  var_indices->operator[](uniqueSize) = Vcopy[i];
	  orbDifference->operator[](uniqueSize) = Ocopy[i];
	}
	i++; uniqueSize++;
      }
      else if (*vec_it < Detcopy[i])
	vec_it ++;
      else {
	vec_it++; i++;
      }
    }

    while (i <Detcopy.size()) {
      Det->operator[](uniqueSize) = Detcopy[i];
      Num->operator[](uniqueSize) = Numcopy[i];
      if (Num2->size() != 0)
	Num2->operator[](uniqueSize) = Num2copy[i];
      if (present->size() != 0)
	present->operator[](uniqueSize) = presentcopy[i];
      Energy->operator[](uniqueSize) = Ecopy[i];
      if (extra_info) {
	var_indices->operator[](uniqueSize) = Vcopy[i];
	orbDifference->operator[](uniqueSize) = Ocopy[i];
      }
      i++; uniqueSize++;
    }

    Det->resize(uniqueSize); Num->resize(uniqueSize);
    if (Num2->size() != 0)
      Num2->resize(uniqueSize);
    if (present->size() != 0)
      present->resize(uniqueSize);//operator[](uniqueSize) = presentcopy->at(i);
    Energy->resize(uniqueSize);
    if (extra_info) {
      var_indices->resize(uniqueSize); orbDifference->resize(uniqueSize);
    }
  }


  void StitchDEH::RemoveDetsPresentIn(Determinant *SortedDets, int DetsSize) {
    int vecid = 0;
    std::vector<Determinant>& Detcopy = *Det;
    std::vector<CItype>& Numcopy = *Num;
    std::vector<CItype>& Num2copy = *Num2;
    std::vector<char>& presentcopy = *present;
    std::vector<double>& Ecopy = *Energy;
    //if (extra_info) {
    std::vector<std::vector<int> >& Vcopy = *var_indices;
    std::vector<std::vector<size_t> >& Ocopy = *orbDifference;
    //}

    size_t uniqueSize = 0, i=0;
    while (i<Detcopy.size() && vecid <DetsSize) {
      if (Detcopy[i] < SortedDets[vecid]) {
	Det->operator[](uniqueSize) = Detcopy[i];
	Num->operator[](uniqueSize) = Numcopy[i];
	if (Num2->size() != 0)
	  Num2->operator[](uniqueSize) = Num2copy[i];
	if (present->size() != 0)
	  present->operator[](uniqueSize) = presentcopy[i];
	Energy->operator[](uniqueSize) = Ecopy[i];
	if (extra_info) {
	  var_indices->operator[](uniqueSize) = Vcopy[i];
	  orbDifference->operator[](uniqueSize) = Ocopy[i];
	}
	i++; uniqueSize++;
      }
      else if (SortedDets[vecid] < Detcopy[i])
	vecid++;
      else {
	vecid++; i++;
      }
    }

    while (i< Detcopy.size()) {
      Det->operator[](uniqueSize) = Detcopy[i];
      Num->operator[](uniqueSize) = Numcopy[i];
      if (Num2->size() != 0)
	Num2->operator[](uniqueSize) = Num2copy[i];
      if (present->size() != 0)
	present->operator[](uniqueSize) = presentcopy[i];
      Energy->operator[](uniqueSize) = Ecopy[i];
      if (extra_info) {
	var_indices->operator[](uniqueSize) = Vcopy[i];
	orbDifference->operator[](uniqueSize) = Ocopy[i];
      }
      i++; uniqueSize++;
    }

    Det->resize(uniqueSize); Num->resize(uniqueSize);
    if (Num2->size() != 0)
      Num2->resize(uniqueSize);
    if (present->size() != 0)
      present->resize(uniqueSize);//operator[](uniqueSize) = presentcopy->at(i);
    Energy->resize(uniqueSize);
    if (extra_info) {
      var_indices->resize(uniqueSize); orbDifference->resize(uniqueSize);
    }
  }


  void StitchDEH::RemoveOnlyDetsPresentIn(Determinant *SortedDets, int DetsSize) {
    int vecid = 0, i=0;
    std::vector<Determinant>& Detcopy = *Det;

    size_t uniqueSize = 0;
    while (i <Detcopy.size() && vecid < DetsSize) {
      if (Detcopy[i] < SortedDets[vecid]) {
	Det->operator[](uniqueSize) = Detcopy[i];
	i++; uniqueSize++;
      }
      else if (SortedDets[vecid] < Detcopy[i])
	vecid++;
      else {
	vecid++; i++;
      }
    }

    while (i <Detcopy.size()) {
      Det->operator[](uniqueSize) = Detcopy[i];
      i++; uniqueSize++;
    }

    Det->resize(uniqueSize); 
  }


  void StitchDEH::RemoveOnlyDetsPresentIn(std::vector<Determinant>& SortedDets) {
    vector<Determinant>::iterator vec_it = SortedDets.begin();
    std::vector<Determinant>& Detcopy = *Det;

    size_t uniqueSize = 0;
    for (size_t i=0; i<Detcopy.size();) {
      if (Detcopy[i] < *vec_it) {
	Det->operator[](uniqueSize) = Detcopy[i];
	i++; uniqueSize++;
      }
      else if (*vec_it < Detcopy[i] && vec_it != SortedDets.end())
	vec_it ++;
      else if (*vec_it < Detcopy[i] && vec_it == SortedDets.end()) {
	Det->operator[](uniqueSize) = Detcopy[i];
	i++; uniqueSize++;
      }
      else {
	vec_it++; i++;
      }
    }
    Det->resize(uniqueSize);
  }

  void StitchDEH::RemoveDuplicates() {
    std::vector<Determinant>& Detcopy = *Det;
    std::vector<CItype>& Numcopy = *Num;
    std::vector<double>& Ecopy = *Energy;
    //if (extra_info) {
    std::vector<std::vector<int> >& Vcopy = *var_indices;
    std::vector<std::vector<size_t> >& Ocopy = *orbDifference;
    //}

    if (Det->size() <= 1) return;
    //if (Det->size() == 1) return;
    size_t uniqueSize = 0;
    for (size_t i=1; i <Detcopy.size(); i++) {
      if (!(Detcopy[i] == Detcopy[i-1])) {
	uniqueSize++;
	Det->operator[](uniqueSize) = Detcopy[i];
	Num->operator[](uniqueSize) = Numcopy[i];
	if (Num2->size() != 0)
	  Num2->operator[](uniqueSize) = Num2->operator[](i);
	if (present->size() != 0)
	  present->operator[](uniqueSize) = present->at(i);
	Energy->operator[](uniqueSize) = Ecopy[i];
	if (extra_info) {
	  var_indices->operator[](uniqueSize) = Vcopy[i];
	  orbDifference->operator[](uniqueSize) = Ocopy[i];
	}
      }
      else {// Same det, so combine
	Num->operator[](uniqueSize) += Numcopy[i];
	if (Num2->size() != 0)
	  Num2->operator[](uniqueSize) += Num2->operator[](i);
	//Num2->operator[](uniqueSize) += Num2copy[i];
	if (present->size() != 0)
	  present->operator[](uniqueSize) = present->at(i);
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
    if (Num2->size() != 0)
      Num2->resize(uniqueSize+1);
    if (present->size() != 0)
      present->resize(uniqueSize+1);//operator[](uniqueSize) = present->at(i);
    Energy->resize(uniqueSize+1);
    if (extra_info) {
      var_indices->resize(uniqueSize+1);
      orbDifference->resize(uniqueSize+1);
    }
  }

  void StitchDEH::deepCopy(const StitchDEH& s) {
    *Det = *(s.Det);
    *Num = *(s.Num);
    if (s.Num2->size() != 0)
      *Num2 = *(s.Num2);
    if (s.present->size() != 0)
      *present = *(s.present);
    *Energy = *(s.Energy);
    if (extra_info) {
      *var_indices = *(s.var_indices);
      *orbDifference = *(s.orbDifference);
    }
  }

  void StitchDEH::operator=(const StitchDEH& s) {
    Det = s.Det;
    Num = s.Num;
    Num2 = s.Num2;
    present = s.present;
    Energy = s.Energy;
    if (extra_info) {
      var_indices = s.var_indices;
      orbDifference = s.orbDifference;
    }
  }

  void StitchDEH::clear() {
    Det->clear();
    Num->clear();
    Num2->clear();
    present->clear();
    Energy->clear();
    if (extra_info) {
      var_indices->clear();
      orbDifference->clear();
    }
  }

  void StitchDEH::resize(size_t s) {
    Det->resize(s);
    Num->resize(s);
    Num2->resize(s);
    present->resize(s);
    Energy->resize(s);
  }


  void StitchDEH::merge(const StitchDEH& s) {
    // Merges with disjoint set
    std::vector<Determinant> Detcopy = *Det;
    std::vector<CItype> Numcopy = *Num;
    std::vector<CItype> Num2copy = *Num2;
    std::vector<char> presentcopy = *present;
    std::vector<double> Ecopy = *Energy;
    //if (extra_info) {
    std::vector<std::vector<int> > Vcopy = *var_indices;
    std::vector<std::vector<size_t> > Ocopy = *orbDifference;
    //}

    Det->resize(Detcopy.size()+s.Det->size());
    Num->resize(Numcopy.size()+s.Det->size());
    Num2->resize(Num2copy.size()+s.Num2->size());
    present->resize(presentcopy.size()+s.present->size());
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
	if (Num2->size() != 0)
	  Num2->operator[](l) = Num2copy.operator[](j);
	if (present->size() != 0)
	  present->operator[](l) = presentcopy.operator[](j);
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
	if (Num2->size() != 0)
	  Num2->operator[](l) = Num2copy.operator[](j);
	if (present->size() != 0)
	  present->operator[](l) = presentcopy.operator[](j);
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
      if (Num2->size() != 0)
	Num2->operator[](l) = Num2copy.operator[](j);
      if (present->size() != 0)
	present->operator[](l) = presentcopy.operator[](j);
      Energy->operator[](l) = Ecopy.operator[](j);
      if (extra_info) {
        var_indices->operator[](l) = Vcopy.operator[](j);
        orbDifference->operator[](l) = Ocopy.operator[](j);
      }
      j++; l++;
    }
    while (k<s.Det->size()) {
      Det->operator[](l) = s.Det->operator[](k);
      Num->operator[](l) = s.Num->operator[](k);
      if (Num2->size() != 0)
	Num2->operator[](l) = Num2copy.operator[](j);
      if (present->size() != 0)
	present->operator[](l) = presentcopy.operator[](j);
      Energy->operator[](l) = s.Energy->operator[](k);
      if (extra_info) {
        var_indices->operator[](l) = s.var_indices->operator[](k);
        orbDifference->operator[](l) = s.orbDifference->operator[](k);
      }
      k++;l++;
    }

  } // merge




};
