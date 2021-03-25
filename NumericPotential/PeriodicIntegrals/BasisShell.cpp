/* Copyright (c) 2012  Gerald Knizia
 * 
 * This file is part of the IR/WMME program
 * (See https://sites.psu.edu/knizia/)
 * 
 * IR/WMME is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 * 
 * IR/WMME is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with bfint (LICENSE). If not, see http://www.gnu.org/licenses/
 */

#include <boost/format.hpp>
#include <ostream>

#include "BasisShell.h"

void BasisSet::basisnormTodensitynorm(int start, int end) {
  for (int i = start ; i < end; i++) {
    BasisShells[i].basisnormTodensitynorm();
  }
}
void BasisSet::densitynormTobasisnorm(int start, int end) {
  for (int i = start ; i < end; i++) {
    BasisShells[i].densitynormTobasisnorm();
  }
}

int BasisSet::getNbas() {
  int nbas = 0;
  for (int i = 0 ; i <BasisShells.size(); i++) {
    nbas += BasisShells[i].nCo * (2 * BasisShells[i].l + 1);
  }
  return nbas;
}

int BasisSet::getNbas(int shlIndex) {
  int nbas = 0;
  for (int i = 0 ; i < shlIndex; i++) {
    nbas += BasisShells[i].nCo * (2 * BasisShells[i].l + 1);
  }
  return nbas;
}

int BasisSet::getNPrimitivebas(int shlIndex) {
  int nbas = 0;
  for (int i = 0 ; i < shlIndex; i++) {
    nbas += BasisShells[i].nFn * (2 * BasisShells[i].l + 1);
  }
  return nbas;
}

void BasisShell::PrintAligned(std::ostream &xout, uint Indent) const
{
   using boost::format;
   
   std::streampos
      p0 = xout.tellp(),
      p1;
//    xout << format("%3i " % iCenter;
   xout << format("%3i: %c   %8.4f %8.4f %8.4f            ")
          % nCo % "spdfghiklm"[l] % Xcoord % Ycoord % Zcoord;
   p1 = xout.tellp();

   // ^- hm... this doesn't work. tellp() I mean.
   p0 = 0;
   p1 = 47;

   for (uint iExp = 0; iExp < exponents.size(); ++iExp){
      if ( iExp != 0 ){
         xout << "\n";
         for ( uint i = 0; i < Indent + p1 - p0; ++ i )
            xout << " ";
      }
      xout << format("%16.7f  ") % exponents[iExp];

      double
         fRenorm = RawGaussNorm(exponents[iExp], l);
      std::stringstream
         str;
      for ( uint iCo = 0; iCo < nCo; ++ iCo ){
         double
             fCo = contractions(iExp, iCo);
         if ( fCo != 0. )
            str << format(" %9.5f") % (fCo*fRenorm);
         else
            str << format(" %9s") % "  - - - -";
      }
      std::string
         s = str.str();
      if (0) {
         while( !s.empty() && (s[s.size()-1] == ' ' || s[s.size()-1] == '-' ) )
            s.resize(s.size() - 1);
      }
      xout << s;
   }
}

double densityNorm(double expl, int l) {
  if (l == 0)
    return pow(M_PI/expl, 1.5);
  else if (l == 1)
    return pow(3.,0.5)*pow(M_PI,1.5)/pow(expl,2.5)/2.;
  else if (l == 2)
    return pow(5.,0.5)*3.*pow(M_PI,1.5)/pow(expl,3.5)/4.;
  else if (l == 3)
    return pow(7.,0.5)*15.*pow(M_PI,1.5)/pow(expl,4.5)/8.;
  else if (l == 4)
    return pow(9.,0.5)*105.*pow(M_PI,1.5)/pow(expl,5.5)/16.;
  else if (l == 5)
    return pow(11.,0.5)*945.*pow(M_PI,1.5)/pow(expl,6.5)/32.;
  else if (l == 6)
    return pow(13.,0.5)*10395.*pow(M_PI,1.5)/pow(expl,7.5)/64.;
}

void BasisShell::basisnormTodensitynorm() {
  //MatrixXd AOnorm(exponents.size(), exponents.size());
  VectorXd DensityNorm(exponents.size());
  for (int i=0; i<contractions.rows(); i++) {
    DensityNorm[i] = densityNorm(exponents[i], l);
    //for (int j=0; j<contractions.rows(); j++) {
    //AOnorm(i,j) = RawGaussProdNorm(exponents[i],l,exponents[j], l);
    //}
  }

  //loop over all contracted gaussians
  for (int j=0; j<contractions.cols(); j++) {
    //double jnorm = contractions.col(j).dot(AOnorm * contractions.col(j));
    double densityNorm = contractions.col(j).dot(DensityNorm);

    for (int i=0; i<contractions.rows(); i++)
      contractions(i,j) /= densityNorm;
  }
}


void BasisShell::densitynormTobasisnorm() {
  MatrixXd AOnorm(exponents.size(), exponents.size());
  VectorXd DensityNorm(exponents.size());
  for (int i=0; i<contractions.rows(); i++) {
    DensityNorm[i] = pow(exponents[i]/M_PI, 1.5);
    for (int j=0; j<contractions.rows(); j++) {
      AOnorm(i,j) = RawGaussProdNorm(exponents[i], l, exponents[j], l);
    }
  }

  //loop over all contracted gaussians
  for (int j=0; j<contractions.cols(); j++) {
    double jnorm = contractions.col(j).dot(AOnorm * contractions.col(j));
    double densityNorm = contractions.col(j).dot(DensityNorm);

    for (int i=0; i<contractions.rows(); i++)
      contractions(i,j) /= sqrt(jnorm);
  }
}

unsigned DoubleFactR(int l) {
   unsigned r = 1;
   while (l > 1) {
      r *= l;
      l -= 2;
   }
   return r;
}

double RawGaussProdNorm(double fExp1, unsigned l1, double fExp2, unsigned l2)
{
  return pow(M_PI/(fExp1+fExp2),1.5) * DoubleFactR(l1+l2-1)/pow(2.*fExp1+2.*fExp2,(l1+l2)/2.);
}

double RawGaussNorm(double fExp, unsigned l)
{
//    return 1./InvRawGaussNorm(fExp, l);
   return pow(M_PI/(2*fExp),.75) * sqrt(DoubleFactR(2*l-1)/pow(4.*fExp,l));
}

