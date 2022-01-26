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
#ifndef SHCI_SORTMPIUTILS_H
#define SHCI_SORTMPIUTILS_H
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <list>
#include <tuple>
#include <map>
#include <boost/serialization/serialization.hpp>

using namespace std;
using namespace Eigen;
class Determinant;
class HalfDet;
class oneInt;
class twoInt;
class twoIntHeatBath;
class twoIntHeatBathSHM;
class schedule;

template <class T> void reorder(vector<T>& A, std::vector<long>& reorder)
{
  vector<T> Acopy = A;
  for (int i=0; i<Acopy.size(); i++) {
    A[i] = Acopy[reorder[i]];
  }
}

template <class T> void reorder(vector<T>& A, std::vector<int>& reorder)
{
  vector<T> Acopy = A;
  for (int i=0; i<Acopy.size(); i++) {
    A[i] = Acopy[reorder[i]];
  }
}

namespace SHCISortMpiUtils {
  int binarySearch(int* arr, int l, int r, int x);

  void RemoveDuplicates(std::vector<Determinant>& Det,
		      std::vector<double>& Num1, std::vector<double>& Num2,
		      std::vector<double>& Energy, std::vector<char>& present);
 void RemoveDetsPresentIn(std::vector<Determinant>& SortedDets, std::vector<Determinant>& Det,
			 std::vector<double>& Num1, std::vector<double>& Num2,
			 std::vector<double>& Energy, std::vector<char>& present) ;
 void RemoveDuplicates(vector<Determinant>& Det) ;



 int partition(Determinant* A, int p,int q, CItype* pNum, double* Energy, vector<double>* det_energy=NULL, vector<char>* present=NULL);
 void quickSort(Determinant* A, int p,int q, CItype* pNum, double* Energy, vector<double>* det_energy=NULL, vector<char>* present=NULL);

 int partitionAll(Determinant* A, int p,int q, CItype* pNum, double* Energy, vector<int>* var_indices, vector<size_t>* orbDifference, vector<double>* det_energy=NULL, vector<bool>* present=NULL);
 void quickSortAll(Determinant* A, int p,int q, CItype* pNum, double* Energy, vector<int>* var_indices, vector<size_t>* orbDifference, vector<double>* det_energy=NULL, vector<bool>* present=NULL);


 void merge(Determinant *a, long low, long high, long mid, long* x, Determinant* c, long* cx);
 void mergesort(Determinant *a, long low, long high, long* x, Determinant* c, long* cx);

 void merge(int *a, long low, long high, long mid, int* x, int* c, int* cx);
 void mergesort(int *a, long low, long high, int* x, int* c, int* cx);

 int ipow(int base, int exp);


 class StitchDEH {
 private:
   friend class boost::serialization::access;
   template<class Archive>
     void serialize(Archive & ar, const unsigned int version) {
     ar & Det & Num & Num2 & present & Energy & var_indices & orbDifference;
   }

 public:
   std::shared_ptr<vector<Determinant> > Det;
   std::shared_ptr<vector<CItype> > Num;       //the numerator for the first Var state
   std::shared_ptr<vector<CItype> > Num2;      //the numberator for the second Var state (if present)
   std::shared_ptr<vector<char> > present;      //the numberator for the second Var state (if present)
   std::shared_ptr<vector<double> > Energy;

   //the next two store information about where a determinant in "C" arose from
   std::shared_ptr<vector<int > > var_indices_beforeMerge;
   std::shared_ptr<vector<size_t > > orbDifference_beforeMerge;

   //the next two store information about where a determinant in "C" arose from,
   //and takes into account the fact that a given determinant might have arisen from
   //multiple parent determinants in "V"
   std::shared_ptr<vector<vector<int> > > var_indices;
   std::shared_ptr<vector<vector<size_t> > > orbDifference;

   bool extra_info; // whether to use var_indices, orbDifference

   StitchDEH();

   StitchDEH(std::shared_ptr<vector<Determinant> >pD,
	     std::shared_ptr<vector<CItype> >pNum,
	     std::shared_ptr<vector<CItype> >pNum2,
	     std::shared_ptr<vector<char> >present,
	     std::shared_ptr<vector<double> >pE,
	     std::shared_ptr<vector<vector<int> > >pvar,
	     std::shared_ptr<vector<vector<size_t> > >porb,
	     std::shared_ptr<vector<int > >pvar_beforeMerge,
	     std::shared_ptr<vector<size_t > >porb_beforeMerge);

   StitchDEH(std::shared_ptr<vector<Determinant> >pD,
	     std::shared_ptr<vector<CItype> >pNum,
	     std::shared_ptr<vector<CItype> >pNum2,
	     std::shared_ptr<vector<char> >present,
	     std::shared_ptr<vector<double> >pE);

   void QuickSortAndRemoveDuplicates() ;
   void MergeSort() ;
   void MergeSortAndRemoveDuplicates() ;
   void RemoveDetsPresentIn(std::vector<Determinant>& SortedDets);
   void RemoveDetsPresentIn(Determinant* SortedDets, int DetsSize);
   void RemoveOnlyDetsPresentIn(Determinant* SortedDets, int DetsSize);
   void RemoveOnlyDetsPresentIn(std::vector<Determinant>& SortedDets) ;
   void RemoveDuplicates();
   void deepCopy(const StitchDEH& s);
   void operator=(const StitchDEH& s);
   void clear();
   void resize(size_t s);
   void merge(const StitchDEH& s);
 };

 /*
 class ElementWiseAddStitchDEH {
 public:
   StitchDEH operator()(const StitchDEH& s1, const StitchDEH& s2) {
     StitchDEH out;
     out.deepCopy(s1);
     out.merge(s2);
     return out;
   }
 };
 */

};

#endif
