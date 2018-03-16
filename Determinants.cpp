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
#include <iostream>
#include <algorithm>
#include "integral.h"
#include "Hmult.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include "Determinants.h"
using namespace std;
using namespace Eigen;



//=============================================================================
void getHijForTReversal(CItype& hij, Determinant& dj, Determinant& dk,
  oneInt& I1, twoInt& I2, double& coreE, size_t& orbDiff, int plusORminus) {
	/*!
	   with treversal symmetry dj and dk will find each other multiple times
	   we prune this possibility as follows
	   ->  we only look for connection is dj is positive (starndard form)
	   -> even it is positive there still might be two connections to dk
	     -> so this updates hij

	   :Arguments:

  	    CItype& hij:
  	        Hamiltonian matrix element, modifed in function.
  	    Determinant& dj:
  	        Determinant j.
  	    Determinant& dk:
  	        Determinant k.
  	    oneInt& I1:
  	        One body integrals.
  	    twoInt& I2:
  	        Two body integrals.
  	    double& coreE:
  	        Core energy.
  	    size_t& orbDiff:
  	        Different number of orbitals between determinants j and k.
  	    int plusORminus:
  	        Currently unused. TODO
	 */
	if (Determinant::Trev != 0 && !dj.hasUnpairedElectrons() &&
	  dk.hasUnpairedElectrons()) {
		Determinant detcpy = dk;
		detcpy.flipAlphaBeta();
		double parity = dk.parityOfFlipAlphaBeta();
		CItype hijCopy = Hij(dj, detcpy, I1, I2, coreE, orbDiff);
		hij = (parity*Determinant::Trev*hijCopy)/pow(2.,0.5);
	}
	else if (Determinant::Trev != 0 && dj.hasUnpairedElectrons() &&
	  !dk.hasUnpairedElectrons()) {
		Determinant detcpy = dj;
		detcpy.flipAlphaBeta();
		double parity = dj.parityOfFlipAlphaBeta();
		CItype hijCopy = Hij(detcpy, dk, I1, I2, coreE, orbDiff);
		hij = (parity*Determinant::Trev*hijCopy)/pow(2.,0.5);
	}
	else if (Determinant::Trev != 0 && dj.hasUnpairedElectrons()
	  && dk.hasUnpairedElectrons()) {
		Determinant detcpyk = dk;
		detcpyk.flipAlphaBeta();
		double parityk = dk.parityOfFlipAlphaBeta();
		CItype hijCopy1 = Hij(dj, detcpyk, I1, I2, coreE, orbDiff);
		hij = Determinant::Trev*parityk*hijCopy1;

	}
}



//=============================================================================
void updateHijForTReversal(CItype& hij, Determinant& dj, Determinant& dk,
  oneInt& I1, twoInt& I2, double& coreE, size_t& orbDiff) {
	/*!
	   with treversal symmetry dj and dk will find each other multiple times
	   we prune this possibility as follows
	   ->  we only look for connection is dj is positive (starndard form)
	   -> even it is positive there still might be two connections to dk
	     -> so this updates hij

	   :Arguments:

  	    CItype& hij:
  	        Hamiltonian matrix element, modifed in function.
  	    Determinant& dj:
  	        Determinant j.
  	    Determinant& dk:
  	        Determinant k.
  	    oneInt& I1:
  	        One body integrals.
  	    twoInt& I2:
  	        Two body integrals.
  	    double& coreE:
  	        Core energy.
  	    size_t& orbDiff:
  	        Different number of orbitals between determinants j and k.
	 */
	if (Determinant::Trev != 0 && !dj.hasUnpairedElectrons() &&
	  dk.hasUnpairedElectrons()) {
		Determinant detcpy = dk;

		detcpy.flipAlphaBeta();
		if (!detcpy.connected(dj)) return;
		double parity = dk.parityOfFlipAlphaBeta();
		CItype hijCopy = Hij(dj, detcpy, I1, I2, coreE, orbDiff);
		hij = (hij + parity*Determinant::Trev*hijCopy)/pow(2.,0.5);
	}
	else if (Determinant::Trev != 0 && dj.hasUnpairedElectrons() &&
	  !dk.hasUnpairedElectrons()) {
		Determinant detcpy = dj;

		detcpy.flipAlphaBeta();
		if (!detcpy.connected(dk)) return;
		double parity = dj.parityOfFlipAlphaBeta();
		CItype hijCopy = Hij(detcpy, dk, I1, I2, coreE, orbDiff);
		hij = (hij + parity*Determinant::Trev*hijCopy)/pow(2.,0.5);
	}
	else if (Determinant::Trev != 0 && dj.hasUnpairedElectrons()
	  && dk.hasUnpairedElectrons()) {
		Determinant detcpyk = dk;

		detcpyk.flipAlphaBeta();
		if (!detcpyk.connected(dj)) return;
		double parityk = dk.parityOfFlipAlphaBeta();
		CItype hijCopy1 = Hij(dj, detcpyk, I1, I2, coreE, orbDiff);
		CItype hijCopy2, hijCopy3;
		hij = hij + Determinant::Trev*parityk*hijCopy1;

	}
}


//=============================================================================
double EnergyAfterExcitation(vector<int>& closed, int& nclosed, oneInt& I1,
  twoInt& I2, double& coreE,  int i, int A, double Energyd) {
	/*!
	   Calculates the new energy of a determinant after single excitation.

	   .. note:: Assumes that the spin of i and a orbitals is the same

	   :Arguments:

  	   vector<int>& closed:
  	       Occupied orbitals in a vector.
  	   int& nclosed:
  	       Number of occupied orbitals.
  	   oneInt& I1:
  	       One body integrals.
  	   twoInt& I2:
  	       Two body integrals.
  	   double& coreE:
  	       Core energy.
  	   int i:
  	       Orbital index for destruction operator.
  	   int A:
  	       Orbital index for creation operator.
  	   double Energyd:
  	       Old determinant energy.

	   :Returns:

  	    double E:
  	        Energy after excitation.
	 */

	double E = Energyd;
#ifdef Complex
	E += -I1(closed[i], closed[i]).real() + I1(A, A).real();
#else
	E += -I1(closed[i], closed[i]) + I1(A, A);
#endif

	for (int I = 0; I<nclosed; I++) {
		if (I == i) continue;
		//E = E - I2.Direct(closed[I]/2, closed[i]/2) + I2.Direct(closed[I]/2, A/2);
		E = E - I2.Direct(closed[I], closed[i]).real() + I2.Direct(closed[I], A).real();
		//if ( (closed[I]%2) == (closed[i]%2) )
		if ( closed[I] == closed[i] )
			//E = E + I2.Exchange(closed[I]/2,closed[i]/2)-I2.Exchange(closed[I]/2,A/2);
			E = E + I2.Exchange(closed[I], closed[i]).real() - I2.Exchange(closed[I], A).real();
	}
	return E;
}

//Assumes that the spin of i and a orbitals is the same
//and the spins of j and b orbitals is the same
//=============================================================================
double EnergyAfterExcitation(vector<int>& closed, int& nclosed, oneInt& I1,
  twoInt& I2, double& coreE,  int i, int A, int j, int B, double Energyd) {
	/*!
	   Calculates the new energy of a determinant after double excitation. i -> A and j -> B.

	   .. note:: Assumes that the spin of each orbital pair (i-A and j-B) is the same.

	   :Arguments:

  	   vector<int>& closed:
  	       Occupied orbitals in a vector.
  	   int& nclosed:
  	       Number of occupied orbitals.
  	   oneInt& I1:
  	       One body integrals.
  	   twoInt& I2:
  	       Two body integrals.
  	   double& coreE:
  	       Core energy.
  	   int i:
  	       Orbital index for destruction operator.
  	   int j:
  	       Orbital index for destruction operator.
  	   int A:
  	       Orbital index for creation operator.
  	   int B:
  	       Orbital index for creation operator.
  	   double Energyd:
  	       Old determinant energy.

	   :Returns:

  	    double E:
  	        Energy after excitation.
	 */

#ifdef Complex
	double E = Energyd - (I1(closed[i], closed[i]) - I1(A, A)+ I1(closed[j],
    closed[j]) - I1(B, B)).real();
#else
	double E = Energyd - I1(closed[i], closed[i]) + I1(A, A)- I1(closed[j],
    closed[j]) + I1(B, B);
#endif

	for (int I = 0; I<nclosed; I++) {
		if (I == i) continue;
		E = E - (I2.Direct(closed[I], closed[i]) - I2.Direct(closed[I], A)).real();
		//if ( (closed[I]%2) == (closed[i]%2) )
		//	E = E + I2.Exchange(closed[I]/2,closed[i]/2)-I2.Exchange(closed[I]/2,A/2);
	}

	for (int I=0; I<nclosed; I++) {
		if (I == i || I == j) continue;
		E = E - (I2.Direct(closed[I], closed[j]) - I2.Direct(closed[I], B)).real();
		//if ( (closed[I]%2) == (closed[j]%2) )
		//	E = E + I2.Exchange(closed[I]/2,closed[j]/2)-I2.Exchange(closed[I]/2,B/2);
	}

	E = E - (I2.Direct(A/2, closed[j]/2) - I2.Direct(A/2, B/2)).real();
	//if ( (closed[i]%2) == (closed[j]%2) )
	//	E = E + I2.Exchange(A/2, closed[j]/2) - I2.Exchange(A/2, B/2);


	return E;
}



//=============================================================================
double Determinant::Energy(oneInt& I1, twoInt&I2, double& coreE) {
	/*!
	   Calculates the energy of the determinant.

	   :Arguments:

  	   oneInt& I1:
  	       One body integrals.
  	   twoInt& I2:
  	       Two body integrals.
  	   double& coreE:
  	       Core energy.

	   :Returns:

  	   double energy+coreE:
  	       Energy of determinant.
	 */

	double energy = 0.0;
	size_t one = 1;
	vector<int> closed;
	for(int i=0; i<EffDetLen; i++) {
		long reprBit = repr[i];
		while (reprBit != 0) {
			int pos = __builtin_ffsl(reprBit);
			closed.push_back(i*64+pos-1);
			reprBit &= ~(one<<(pos-1));
		}
	}

	for (int i=0; i<closed.size(); i++) {
		int I = closed.at(i);
#ifdef Complex
		energy += I1(I,I).real();
#else
		energy += I1(I,I);
#endif
		for (int j=i+1; j<closed.size(); j++) {
			int J = closed.at(j);
			//energy += I2.Direct(I/2,J/2);
			energy += I2.Direct(I,J).real();
			if ( (I%2) == (J%2) ) {
				//energy -= I2.Exchange(I/2, J/2);
				energy -= I2.Exchange(I,J).real();
			}
		}
	}

	return energy+coreE;
}



//=============================================================================
void Determinant::initLexicalOrder(int nelec) {
	LexicalOrder.setZero(norbs-nelec+1, nelec);
	Matrix<size_t, Dynamic, Dynamic> NodeWts(norbs-nelec+2, nelec+1);
	NodeWts(0,0) = 1;
	for (int i=0; i<nelec+1; i++)
		NodeWts(0,i) = 1;
	for (int i=0; i<norbs-nelec+2; i++)
		NodeWts(i,0) = 1;

	for (int i=1; i<norbs-nelec+2; i++)
		for (int j=1; j<nelec+1; j++)
			NodeWts(i,j) = NodeWts(i-1, j) + NodeWts(i, j-1);

	for (int i=0; i<norbs-nelec+1; i++) {
		for (int j=0; j<nelec; j++) {
			LexicalOrder(i,j) = NodeWts(i,j+1)-NodeWts(i,j);
		}
	}
}

//=============================================================================
double parity(char* d, int& sizeA, int& i) {
	double sgn = 1.;
	for (int j=0; i<sizeA; j++) {
		if (j >= i)
			break;
		if (d[j] != 0)
			sgn *= -1;
	}
	return sgn;
}



//=============================================================================
void Determinant::parity(int& i, int& j, int& a, int& b, double& sgn) {
  /*!
  Calculates the parity of the double excitation operator on the determinant. Where i -> a and j -> b, i.e. :math:`\Gamma = a^\dagger_i a^\dagger_j a_b a_a`

  :Arguments:

      int& i:
          Creation operator index.
      int& j:
          Creation operator index.
      int& a:
          Destruction operator index.
      int& b:
          Destruction operator index.
      double& sgn:
          Parity, modified in function.
  */
	parity(min(i, a), max(i,a), sgn);
	setocc(i, false); setocc(a,true);
	parity(min(j, b), max(j,b), sgn);
	setocc(i, true); setocc(a, false);
	return;
}



//=============================================================================
CItype Determinant::Hij_2Excite(int& i, int& j, int& a, int& b, oneInt&I1,
  twoInt& I2) {
  /*!
  Calculate the hamiltonian matrix element connecting determinants connected by  :math:`\Gamma = a^\dagger_i a^\dagger_j a_b a_a`, i.e. double excitation.

  :Arguments:

      int& i:
          Creation operator index.
      int& j:
          Creation operator index.
      int& a:
          Destruction operator index.
      int& b:
          Destruction operator index.
      oneInt& I1:
          One body integrals.
      twoInt& I2:
          Two body integrals.

  */
	double sgn = 1.0;
	int I = min(i,j), J= max(i,j), A= min(a,b), B = max(a,b);
	parity(min(I, A), max(I,A), sgn);
	parity(min(J, B), max(J,B), sgn);
	if(A>J || B<I) sgn *= -1.;
	return sgn*(I2(A,I,B,J) - I2(A,J,B,I));
}


//=============================================================================
CItype Hij_1Excite(int a, int i, oneInt& I1, twoInt& I2, int* closed,
  int& nclosed) {
	//int a = cre[0], i = des[0];
  /*!
  Calculate the hamiltonian matrix element connecting determinants connected by  :math:`\Gamma = a^\dagger_a a_i`, i.e. single excitation.

  .. note:: This version of the function is not a member function of the determinant class.

  :Arguments:

      int& a:
          Creation operator index.
      int& i:
          Destruction operator index.
      oneInt& I1:
          One body integrals.
      twoInt& I2:
          Two body integrals.
      int* closed:
          Vector of indices for occupied orbitals.
      int& nclosed:
          Number of occupied orbitals.

  */
	double sgn=1.0;

	CItype energy = I1(a,i);
	for (int j=0; j<nclosed; j++) {
		if (closed[j]>min(i,a)&& closed[j] <max(i,a))
			sgn*=-1.;
		energy += (I2(a,i,closed[j],closed[j]) - I2(a,closed[j],closed[j], i));
	}

	return energy*sgn;
}


//=============================================================================
CItype Determinant::Hij_1Excite(int& a, int& i, oneInt&I1, twoInt& I2) {
  /*!
  Calculate the hamiltonian matrix element connecting determinants connected by  :math:`\Gamma = a^\dagger_a a_i`, i.e. single excitation.

  :Arguments:

      int& a:
          Creation operator index.
      int& i:
          Destruction operator index.
      oneInt& I1:
          One body integrals.
      twoInt& I2:
          Two body integrals.

  */
	double sgn = 1.0;
	parity(min(a,i), max(a,i), sgn);

	CItype energy = I1(a,i);
	long one = 1;
	for (int I=0; I<EffDetLen; I++) {

		long reprBit = repr[I];
		while (reprBit != 0) {
			int pos = __builtin_ffsl(reprBit);
			int j = I*64+pos-1;
			energy += (I2(a,i,j,j) - I2(a,j,j,i));
			reprBit &= ~(one<<(pos-1));
		}

	}
	energy *= sgn;
	return energy;
}


//=============================================================================
void getOrbDiff(Determinant& bra, Determinant &ket, size_t &orbDiff) {
  /*!
  Calculates the number of orbitals with differing occuations between bra and ket.

  :Arguments:

      Determinant& bra:
          Determinant in bra.
      Determinant& ket:
          Determinant in ket.
      size_t& orbDiff:
          Number of orbitals with differing occupations. Changed in this function.
  */
	int cre[2],des[2],ncre=0,ndes=0; long u,b,k,one=1;
	cre[0]=-1; cre[1]=-1; des[0]=-1; des[1]=-1;

	for (int i=0; i<Determinant::EffDetLen; i++) {
		u = bra.repr[i] ^ ket.repr[i];
		b = u & bra.repr[i]; //the cre bits
		k = u & ket.repr[i]; //the des bits

		while(b != 0) {
			int pos = __builtin_ffsl(b);
			cre[ncre] = pos-1+i*64;
			ncre++;
			b &= ~(one<<(pos-1));
		}
		while(k != 0) {
			int pos = __builtin_ffsl(k);
			des[ndes] = pos-1+i*64;
			ndes++;
			k &= ~(one<<(pos-1));
		}
	}

	if (ncre == 0) {
		orbDiff = 0;
	}
	else if (ncre ==1 ) {
		size_t c0=cre[0], N=bra.norbs, d0 = des[0];
		orbDiff = c0*N+d0;
	}
	else if (ncre == 2) {
		size_t c0=cre[0], c1=cre[1], d1=des[1],N=bra.norbs, d0 = des[0];
		orbDiff = c1*N*N*N+d1*N*N+c0*N+d0;
	}
	else {
		cout << "Different greater than 2."<<endl;
		exit(0);
	}

}

//=============================================================================
CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2,
  double& coreE, size_t& orbDiff) {
    /*!
    Calculates the hamiltonian matrix element connecting the two determinants bra and ket.

    :Arguments:

        Determinant& bra:
            Determinant in bra.
        Determinant& ket:
            Determinant in ket.
        oneInt& I1:
           One body integrals.
        twoInt& I2:
           Two body integrals.
        double& coreE:
           Core energy.
        size_t& orbDiff:
            Number of orbitals with differing occupations.
    */
	int cre[200],des[200],ncre=0,ndes=0; long u,b,k,one=1;
	cre[0]=-1; cre[1]=-1; des[0]=-1; des[1]=-1;

	for (int i=0; i<Determinant::EffDetLen; i++) {
		u = bra.repr[i] ^ ket.repr[i];
		b = u & bra.repr[i]; //the cre bits
		k = u & ket.repr[i]; //the des bits

		while(b != 0) {
			int pos = __builtin_ffsl(b);
			cre[ncre] = pos-1+i*64;
			ncre++;
			b &= ~(one<<(pos-1));
		}
		while(k != 0) {
			int pos = __builtin_ffsl(k);
			des[ndes] = pos-1+i*64;
			ndes++;
			k &= ~(one<<(pos-1));
		}
	}

	if (ncre == 0) {
		cout << bra<<endl;
		cout << ket<<endl;
		cout <<"Use the function for energy"<<endl;
		exit(0);
	}
	else if (ncre ==1 ) {
		size_t c0=cre[0], N=bra.norbs, d0 = des[0];
		orbDiff = c0*N+d0;
		//orbDiff = cre[0]*bra.norbs+des[0];
		return ket.Hij_1Excite(cre[0], des[0], I1, I2);
	}
	else if (ncre == 2) {
		size_t c0=cre[0], c1=cre[1], d1=des[1],N=bra.norbs, d0 = des[0];
		orbDiff = c1*N*N*N+d1*N*N+c0*N+d0;
		//orbDiff = cre[1]*bra.norbs*bra.norbs*bra.norbs+des[1]*bra.norbs*bra.norbs+cre[0]*bra.norbs+des[0];
		return ket.Hij_2Excite(des[0], des[1], cre[0], cre[1], I1, I2);
	}
	else {
		//cout << "Should not be here"<<endl;
		return 0.;
	}
}
