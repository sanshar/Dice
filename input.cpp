#include "global.h"
#include "input.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace boost;

void readInput(string input, std::vector<int>& occupied, schedule& schd) {
  ifstream dump(input.c_str());
  int maxiter = -1;
  vector<int> sweep_iter;
  vector<double> sweep_epsilon;
  schd.nblocks = 1;
  schd.restart = false;
  schd.fullrestart = false;
  schd.davidsonTol = 5.e-5;
  schd.epsilon2 = 1.e-8;
  schd.epsilon2Large = 1000.0;
  schd.dE = 1.e-8;
  schd.prefix = ".";
  schd.stochastic = true;
  schd.SampleN = -1;
  schd.excitation = 1000;
  schd.nvirt = 1e6;
  schd.onlyperturbative = false;
  schd.singleList = true;
  int nocc = -1;;
  schd.io = true;

  while (dump.good()) {

    std::string
      Line;
    std::getline(dump, Line);
    cout << "#"<<Line<<endl;

    vector<string> tok;
    boost::split(tok, Line, is_any_of(", \t"), token_compress_on);
    string ArgName = *tok.begin();

    if (dump.eof())
      break;
    if (!ArgName.empty() && (boost::iequals(tok[0].substr(0,1), "#"))) continue;
    if (ArgName.empty()) continue;

    if (boost::iequals(ArgName, "nocc")) {
      nocc = atoi(tok[1].c_str());
      occupied.resize(nocc);
      cout << "#";
      for (int i=0; i<nocc; i++) {
	dump >> occupied[i];
	cout << occupied[i]<<" ";
      }
      cout <<endl;
    }
    else if (boost::iequals(ArgName, "noio")) 
      schd.io=false;
    else if (boost::iequals(ArgName, "epsilon2")) 
      schd.epsilon2 = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "epsilon2Large")) 
      schd.epsilon2Large = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "onlyperturbative")) 
      schd.onlyperturbative = true;
    else if (boost::iequals(ArgName, "nblocks")) 
      schd.nblocks = atof(tok[1].c_str());
    else if (boost::iequals(ArgName , "davidsonTol")) 
      schd.davidsonTol = atof(tok[1].c_str());
    else if (boost::iequals(ArgName , "excitation")) 
      schd.excitation = atof(tok[1].c_str());
    else if (boost::iequals(ArgName , "sampleN")) 
      schd.SampleN = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "restart")) 
      schd.restart = true;
    else if (boost::iequals(ArgName, "nvirt")) 
      schd.nvirt = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "deterministic")) 
      schd.stochastic = false;
    else if (boost::iequals(ArgName, "fullrestart")) 
      schd.fullrestart = true;
    else if (boost::iequals(ArgName, "dE") )
      schd.dE = atof(tok[1].c_str());
    else if (boost::iequals(ArgName , "eps") )
      schd.eps = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "prefix") )
      schd.prefix = tok[1].c_str();
    else if (boost::iequals(ArgName, "schedule")) { 

      std::getline(dump, Line);
      cout << "#"<<Line<<endl;
      vector<string> schd_tok;
      boost::split(schd_tok, Line, is_any_of(" \t"), token_compress_on);
      while(!boost::iequals(schd_tok[0], "END")) {
	if (!boost::iequals(schd_tok[0].substr(0,1), "#")) {
	  sweep_iter.push_back( atoi(schd_tok[0].c_str()));
	  sweep_epsilon.push_back( atof(schd_tok[1].c_str()));
	}
	std::getline(dump, Line);
	cout << "#"<<Line<<endl;
	boost::split(schd_tok, Line, is_any_of(" \t"), token_compress_on);
      }
    }
    else if (boost::iequals(ArgName, "maxiter")) 
      maxiter = atoi(tok[1].c_str());
    else {
      cout << "cannot read option "<<ArgName<<endl;
      exit(0);
    }
  }

  if (maxiter < sweep_iter[sweep_iter.size()-1]) {
    cout << "maxiter should be greater than last entry of sweep_iter"<<endl;
    exit(0);
  }
  if (nocc == -1) {
    cout << "nocc keyword has to be included."<<endl;
    exit(0);
  }
  for (int i=1; i<sweep_iter.size(); i++) 
    for (int j=sweep_iter[i-1]; j<sweep_iter[i]; j++)
      schd.epsilon1.push_back(sweep_epsilon[i-1]);

  for (int j=sweep_iter[sweep_iter.size()-1]; j<maxiter; j++)
    schd.epsilon1.push_back(sweep_epsilon[sweep_iter.size()-1]);
}

