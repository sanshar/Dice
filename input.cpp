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
#include "input.h"
#include "CPS.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

using namespace Eigen;
using namespace boost;
using namespace std;

void readCorrelator(std::string input, int correlatorSize,
		    std::vector<CPS>& correlators) {
  ifstream dump(input.c_str());

  while (dump.good()) {

    std::string
      Line;
    std::getline(dump, Line);
    trim(Line);
    vector<string> tok;
    boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);

    string ArgName = *tok.begin();

    //if (dump.eof())
    //break;
    if (!ArgName.empty() && (boost::iequals(tok[0].substr(0,1), "#"))) continue;
    if (ArgName.empty()) continue;
    
    if (tok.size() != correlatorSize) {
      cout << "Something wrong in line : "<<Line<<endl;
      exit(0);
    }

    vector<int> asites, bsites;
    for (int i=0; i<correlatorSize; i++) {
      int site = atoi(tok[i].c_str());
      if (site%2 == 0) asites.push_back(site/2);
      else             bsites.push_back(site/2);
    }
    correlators.push_back(CPS(asites, bsites));
  }
}


void readHF(MatrixXd& Hfmatrix) {

  ifstream dump("hf.txt");
  for (int i=0; i<Hfmatrix.rows(); i++)
  for (int j=0; j<Hfmatrix.rows(); j++)
    dump >> Hfmatrix(i,j);
}

void readInput(string input, schedule& schd) {
  cout << "**************************************************************"<<endl;
  cout << "Input file  :"<<endl;
  cout << "**************************************************************"<<endl;

  ifstream dump(input.c_str());

  schd.deterministic = false;
  schd.restart       = false;

  while (dump.good()) {

    std::string
      Line;
    std::getline(dump, Line);
    trim(Line);
    cout <<Line<<endl;

    vector<string> tok;
    boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);
    string ArgName = *tok.begin();

    //if (dump.eof())
    //break;
    if (!ArgName.empty() && (boost::iequals(tok[0].substr(0,1), "#"))) continue;
    if (ArgName.empty()) continue;

    if (boost::iequals(ArgName, "restart"))
      schd.restart = true;
    else if (boost::iequals(ArgName, "deterministic"))
      schd.deterministic = true;
    else if (boost::iequals(ArgName, "tol"))
      schd.tol = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "correlator")) {
      int siteSize = atoi(tok[1].c_str());
      schd.correlatorFiles[siteSize] = tok[2];
    }
    else {
      cout << "cannot read option "<<ArgName<<endl;
      exit(0);
    }

  }


}
