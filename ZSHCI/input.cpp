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

void readInput(string input, std::vector<std::vector<int> >& occupied, schedule& schd) {
  cout << endl;
  cout << endl;
  cout << "**************************************************************" << endl;
  cout << "INPUT FILE" << endl;
  cout << "**************************************************************" << endl;

  ifstream dump(input.c_str());
  int maxiter = -1;
  vector<int> sweep_iter;
  vector<double> sweep_epsilon;
  int nocc = -1;

  schd.thresh_hij = 1e-10;
  schd.davidsonTol = 5.e-5;
  schd.davidsonTolLoose = 5.e-5;
  schd.RdmType = RELAXED;
  schd.DavidsonType = MEMORY;
  schd.epsilon2 = 1.e-8;
  schd.epsilon2Large = 1000.0;
  schd.SampleN = -1;

  schd.onlyperturbative = false;
  schd.restart = false;
  schd.fullrestart = false;
  schd.dE = 1.e-8;


  schd.stochastic = true;
  schd.nblocks = 1;
  schd.excitation = 1000;
  schd.nvirt = 1e6;
  schd.singleList = true;
  schd.io = true;
  schd.nroots = 1;
  schd.nPTiter = 10000;
  schd.DoRDM = false;
  schd.DoSpinRDM = false;
  schd.quasiQ = false;
  schd.doSOC = false;
  schd.doSOCQDPT = false;
  schd.randomSeed = getTime();
  schd.doGtensor = false;
  schd.integralFile = "FCIDUMP";
  schd.doResponse = false;
  schd.responseFile = "RESPONSE";
  schd.socmultiplier = 1.0;
  schd.targetError = 1.e-4;
  schd.num_thrds = 1;
  schd.Trev = 0;
  schd.algorithm = 0;
  schd.outputlevel = 0;
  schd.printBestDeterminants = 0;
  schd.writeBestDeterminants = 0;
  schd.writeBestDeterminantsUHF = 0;
  schd.extrapolate = false;
  schd.extrapolationFactor = 2.0/3.0;
  schd.enforceSeniority = false;
  schd.maxSeniority = 10000;
  schd.enforceExcitation = false;
  schd.maxExcitation = 10000;
  schd.enforceSenioExc = false;
  schd.ncore = 0;
  //the ridiculously large number of active spacce orbitals
  schd.nact = 1000000;
  schd.doLCC = false;
  schd.DoOneRDM = false;
  schd.DoSpinOneRDM = false;

  schd.pointGroup = "c1";
  schd.spin = 0;
  schd.irrep = 1;
  schd.DoOneRDM = false;
  schd.DoThreeRDM = false;
  schd.DoFourRDM = false;

  schd.ReadTxt = false;

  schd.cdfciIter = 0;
  schd.report_interval = 1000;
  schd.z_threshold = 0.0;
  schd.max_determinants = 10000000;
  schd.sampleNewDets = false;
  schd.precondition = false;
  
  schd.jz = -1000; // if jz is -1000 indicating no jz symmetry.

  while (dump.good()) {

    std::string Line;
    std::getline(dump, Line);
    trim(Line);
    cout << Line << endl;

    vector<string> tok;
    boost::split(tok, Line, is_any_of(", \t\n"), token_compress_on);
    string ArgName = *tok.begin();

    //if (dump.eof())
    //break;
    if (!ArgName.empty() && (boost::iequals(tok[0].substr(0,1), "#"))) continue;
    if (ArgName.empty()) continue;

    if (boost::iequals(ArgName, "nocc")) {
      nocc = atoi(tok[1].c_str());

      std::string Line;
      vector<string> tok;

      std::getline(dump, Line);
      trim(Line);
      boost::split(tok, Line, is_any_of(", \t"), token_compress_on);
      int index =0;
      while (!boost::iequals(tok[0], "end")) {

        occupied.push_back(vector<int>(nocc));
        if (nocc != tok.size()) {
          cout << "nocc: " << nocc << " neq " << tok.size() << endl;
          for (int t=0; t<tok.size(); t++)
            cout << tok[t] << "  ";
          exit(0);
        }

        for (int i=0; i<tok.size(); i++) {
          occupied[index][i] = atoi(tok[i].c_str());
          cout << occupied[index][i] << " ";
        }
        cout << endl;
        std::getline(dump, Line);
        trim(Line);
        boost::split(tok, Line, is_any_of(", \t"), token_compress_on);
        index++;
      }
    }
    else if (boost::iequals(ArgName, "nact"))
      schd.nact = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "ncore"))
      schd.ncore = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "noio"))
      schd.io=false;
    else if (boost::iequals(ArgName, "dolcc"))
      schd.doLCC=true;
    else if (boost::iequals(ArgName, "io"))
      schd.io=true;
    else if (boost::iequals(ArgName, "directdavidson"))
      schd.DavidsonType = DIRECT;
    else if (boost::iequals(ArgName, "diskdavidson"))
      schd.DavidsonType = DISK;
    else if (boost::iequals(ArgName, "relaxedRDM"))
      schd.RdmType = UNRELAXED;
    else if (boost::iequals(ArgName, "num_thrds"))
      schd.num_thrds = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "outputlevel"))
      schd.outputlevel = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "extrapolate")) {
      schd.extrapolate = true;
      if (tok.size() == 2)
        schd.extrapolationFactor = atoi(tok[1].c_str());
    }
    else if (boost::iequals(ArgName, "dosoc"))
      schd.doSOC=true;
    else if (boost::iequals(ArgName, "algorithm"))
      schd.algorithm=atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "doresponse"))  {
      schd.doResponse=true;
      schd.responseFile = tok[1];
    }
    else if (boost::iequals(ArgName, "maxseniority")) {
      schd.enforceSeniority = true;
      if (tok.size() == 1)
        schd.maxSeniority = 0;
      else
        schd.maxSeniority = atoi(tok[1].c_str());
    }
    else if (boost::iequals(ArgName, "maxexcitation")) {
      schd.enforceExcitation = true;
      if (tok.size() == 1)
        schd.maxExcitation = 0;
      else
        schd.maxExcitation = atoi(tok[1].c_str());
    }
    else if (boost::iequals(ArgName, "SenioAndExc"))
      schd.enforceSenioExc = true;
    else if (boost::iequals(ArgName, "dogtensor"))
      schd.doGtensor=true;
    else if (boost::iequals(ArgName, "targetError"))
      schd.targetError=atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "orbitals"))
      schd.integralFile = tok[1];
    else if (boost::iequals(ArgName, "dosocqdpt"))
      schd.doSOCQDPT=true;
    else if (boost::iequals(ArgName, "nptiter"))
      schd.nPTiter=atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "epsilon2"))
      schd.epsilon2 = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "socmultiplier"))
      schd.socmultiplier = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "seed"))
      schd.randomSeed = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "nroots"))
      schd.nroots = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "epsilon2Large"))
      schd.epsilon2Large = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "onlyperturbative"))
      schd.onlyperturbative = true;
    else if (boost::iequals(ArgName, "printbestdeterminants"))
      schd.printBestDeterminants = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "writebestdeterminants"))
      schd.writeBestDeterminants = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "writebestdeterminantsuhf"))
      schd.writeBestDeterminantsUHF = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "dordm"))
      schd.DoRDM = true;
    else if (boost::iequals(ArgName, "DoOneRDM"))
      schd.DoOneRDM = true;
    else if (boost::iequals(ArgName, "DoSpinOneRDM"))
      schd.DoSpinOneRDM = true;
    else if (boost::iequals(ArgName, "Treversal")) {
      schd.Trev = atoi(tok[1].c_str());
      if (!(schd.Trev == 0 || schd.Trev == 1 || schd.Trev == -1)) {
        cout << "Treversal should be either 0, 1, or -1." << endl;
        exit(0);
      }
    }
    else if (boost::iequals(ArgName, "dospinrdm")) {
      schd.DoRDM = true;
      schd.DoSpinRDM = true;
    }
    else if (boost::iequals(ArgName, "quasiq")) {
      schd.quasiQ = true;
      schd.quasiQEpsilon = atof(tok[1].c_str());
    }
    else if (boost::iequals(ArgName, "nblocks"))
      schd.nblocks = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "thresh_hij"))
      schd.thresh_hij = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "davidsonTol"))
      schd.davidsonTol = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "davidsonTolLoose"))
      schd.davidsonTolLoose = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "excitation"))
      schd.excitation = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "sampleN"))
      schd.SampleN = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "restart"))
      schd.restart = true;
    else if (boost::iequals(ArgName, "nvirt"))
      schd.nvirt = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "deterministic"))
      schd.stochastic = false;
    else if (boost::iequals(ArgName, "fullrestart"))
      schd.fullrestart = true;
    else if (boost::iequals(ArgName, "dE"))
      schd.dE = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "eps"))
      schd.eps = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "prefix"))
      schd.prefix.push_back(tok[1]);
    else if (boost::iequals(ArgName, "pointGroup") )
      schd.pointGroup = tok[1];
    else if (boost::iequals(ArgName, "spin"))
      schd.spin = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "irrep"))
      schd.irrep = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "DoOneRDM"))
      schd.DoOneRDM = true;
    else if (boost::iequals(ArgName, "DoThreeRDM"))
      schd.DoThreeRDM = true;
    else if (boost::iequals(ArgName, "DoFourRDM"))
      schd.DoFourRDM = true;
    else if (boost::iequals(ArgName, "readText"))
      schd.ReadTxt = true;
    else if (boost::iequals(ArgName, "schedule")) {
      std::getline(dump, Line);
      cout << Line << endl;
      vector<string> schd_tok;
      boost::split(schd_tok, Line, is_any_of(" \t"), token_compress_on);
      while (!boost::iequals(schd_tok[0], "END")) {
        if (!boost::iequals(schd_tok[0].substr(0, 1), "#")) {
          sweep_iter.push_back(atoi(schd_tok[0].c_str()));
          sweep_epsilon.push_back(atof(schd_tok[1].c_str()));
        }
        std::getline(dump, Line);
        cout << Line << endl;
        boost::split(schd_tok, Line, is_any_of(" \t"), token_compress_on);
      }
    }
    else if (boost::iequals(ArgName, "maxiter"))
      maxiter = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "cdfciIter"))
      schd.cdfciIter = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "cdfciOn"))
      schd.cdfci_on = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "cdfciTol"))
      schd.cdfciTol = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "reportInterval"))
      schd.report_interval = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "z_threshold"))
      schd.z_threshold = atof(tok[1].c_str());
    else if (boost::iequals(ArgName, "cdfciSample"))
      schd.sampleNewDets = true;
    else if (boost::iequals(ArgName, "cdfci_precondition"))
      schd.precondition = true;
    else if (boost::iequals(ArgName, "max_determinants"))
      schd.max_determinants = atoi(tok[1].c_str());
    else if (boost::iequals(ArgName, "double_group"))
      schd.double_group = tok[1];
    else if (boost::iequals(ArgName, "jz"))
      schd.jz = atoi(tok[1].c_str()); 
    else {
      cout << "cannot read option " << ArgName << endl;
      exit(0);
    }
  }

  if (maxiter < sweep_iter[sweep_iter.size() - 1]) {
    cout << "maxiter should be greater than last entry of sweep_iter" << endl;
    exit(0);
  }
  if (nocc == -1) {
    cout << "nocc keyword has to be included." << endl;
    exit(0);
  }
  //if (schd.DavidsonType == DIRECT)
  //  schd.davidsonTolLoose = 3.e-2;

  for (int i=1; i<sweep_iter.size(); i++)
    for (int j=sweep_iter[i-1]; j<sweep_iter[i]; j++)
      schd.epsilon1.push_back(sweep_epsilon[i-1]);

  for (int j=sweep_iter[sweep_iter.size()-1]; j<maxiter; j++)
    schd.epsilon1.push_back(sweep_epsilon[sweep_iter.size()-1]);

  if (schd.prefix.size() == 0)
    schd.prefix.push_back(".");

  //cout << "**************************************************************" << endl;
}
