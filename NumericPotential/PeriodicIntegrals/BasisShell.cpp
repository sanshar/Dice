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

unsigned DoubleFactR(int l) {
   unsigned r = 1;
   while (l > 1) {
      r *= l;
      l -= 2;
   }
   return r;
}

double RawGaussNorm(double fExp, unsigned l)
{
//    return 1./InvRawGaussNorm(fExp, l);
   return pow(M_PI/(2*fExp),.75) * sqrt(DoubleFactR(2*l-1)/pow(4.*fExp,l));
}

