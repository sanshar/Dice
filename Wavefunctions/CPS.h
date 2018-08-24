/*
  Developed by Sandeep Sharma
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

#ifndef CPS_HEADER_H
#define CPS_HEADER_H
#include "input.h"
#include "Correlator.h"
#include "Determinants.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>


/*
 * CPS is a product of correlators 
 */
class CPS {
    private:
        friend class boost::serialization::access;
        template<class Archive>
            void serialize (Archive & ar, const unsigned int version) {
                ar & cpsArray
                   & orbitalToCPS
                   & workingVectorOfCPS;
            }
    public:
        std::vector<Correlator> cpsArray;  
        vector<vector<int>> orbitalToCPS; //for each orbital all coorelators that it belongs to
        vector<int> workingVectorOfCPS;

        //reads correlator file and makes cpsArray, orbitalToCPS
        CPS () {    
            for (auto it = schd.correlatorFiles.begin(); it != schd.correlatorFiles.end(); it++) {
                readCorrelator(it->second, it->first, cpsArray);
            }
            
            int maxCPSSize = 0;
            orbitalToCPS.resize(Determinant::norbs);
            for (int i = 0; i < cpsArray.size(); i++) {
                for (int j = 0; j < cpsArray[i].asites.size(); j++)
                    orbitalToCPS[cpsArray[i].asites[j]].push_back(i);
            }
            for (int i = 0; i < orbitalToCPS.size(); i++)
              if (orbitalToCPS[i].size() > maxCPSSize)
                maxCPSSize = orbitalToCPS[i].size();
            workingVectorOfCPS.resize(4 * maxCPSSize);
        };

        /*
         * construct using correlator vector
         */
        CPS (std::vector<Correlator>& pcpsArray) : cpsArray(pcpsArray) {
            orbitalToCPS.resize(Determinant::norbs);
            for (int i = 0; i < cpsArray.size(); i++)
            {
              for (int j = 0; j < cpsArray[i].asites.size(); j++)
                orbitalToCPS[cpsArray[i].asites[j]].push_back(i);
            } 
            
            int maxCPSSize = 0;
            orbitalToCPS.resize(Determinant::norbs);
            for (int i = 0; i < cpsArray.size(); i++) {
                for (int j = 0; j < cpsArray[i].asites.size(); j++)
                    orbitalToCPS[cpsArray[i].asites[j]].push_back(i);
            }
            for (int i = 0; i < orbitalToCPS.size(); i++)
              if (orbitalToCPS[i].size() > maxCPSSize)
                maxCPSSize = orbitalToCPS[i].size();
            workingVectorOfCPS.resize(4 * maxCPSSize);
        };

        /*
        * Takes an occupation number representation of a determinant
        * in the local orbital basis and calculates the overlap with
        * the CPS
        * PARAMS:
        * 
        * Determinant: the occupation number representation as an input
        * 
        * RETURN:
        * <d|CPS>
        */
         double Overlap(const Determinant& d);

        /*
        * Takes an occupation number representation of two determinants
        * in the local orbital basis and calculates the ratio of overlaps 
        * <d1|CPS>/<d2|CPS>
        * PARAMS:
        * 
        * Determinant: the occupation number representation of dets
        * 
        * RETURN:
        * <d1|CPS>/<d2|CPS>
        *
        */
         double OverlapRatio(const Determinant& d1, const Determinant& d2);

        /*
         * return ratio of overlaps of CPS with d and (i->a,j->b)excited-d (=dcopy)
         */
        double OverlapRatio(int i, int a, Determinant &dcopy, Determinant &d);
        double OverlapRatio(int i, int j, int a, int b, Determinant &dcopy, Determinant &d);

        /*
        * Takes an occupation number representation of a determinant
        * in the local orbital basis and calculates the overlap 
        * the CPS and also the overlap of the determinant
        * with respect to the tangent vector w.r.t to all the 
        * parameters in the wavefuncion
        * 
        * PARAMS:
        * 
        * Determinant: the occupation number representation of the determinant
        * grad       : the vector of the gradient. This vector is long and contains
        *              the space for storing gradients with respect to all the parameters
        *              in the wavefunction and not just this CPS and so the index startIndex
        *              is needed to indentify at what index should we start populating the
        *              gradient w.r.t to this CPS in the vector
        * ovlp       : <d|CPS>
        * startIndex : The location in the vector gradient from where to start populating
        *              the gradient w.r.t to the current Correlator
        */
         void   OverlapWithGradient  (const Determinant& d, 
           		                         Eigen::VectorXd& grad,
                             			   const double& ovlp);

         void getVariables(Eigen::VectorXd &v);
         long getNumVariables();
         void updateVariables(Eigen::VectorXd &v);
         void printVariables();
};


#endif
