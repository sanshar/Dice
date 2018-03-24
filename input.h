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
#ifndef INPUT_HEADER_H
#define INPUT_HEADER_H
#include <Eigen/Dense>
#include <string>
#include <map>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>

class CPS;

void readHF(Eigen::MatrixXd&);

struct schedule {
private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & restart & deterministic
      & tol & correlatorFiles;
  }
public:
  bool restart;
  bool deterministic;
  double tol;
  std::map<int, std::string> correlatorFiles;
};

void readInput(std::string input, schedule& schd);
void readCorrelator(std::string input, int correlatorSize,
		    std::vector<CPS>& correlators);
#endif
