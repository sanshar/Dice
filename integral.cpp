#include "integral.h"
#include <fstream>
#include "string.h"
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include "math.h"

using namespace boost;
bool myfn(double i, double j) { return fabs(i)<fabs(j); }
void readIntegrals(string fcidump, twoInt& I2, oneInt& I1, int& nelec, int& norbs, double& coreE,
		   std::vector<int>& irrep) {

  ifstream dump(fcidump.c_str());
  bool startScaling = false;
  norbs = -1;
  nelec = -1;
    
  int index = 0;
  vector<string> tok;
  string msg;
  while(!dump.eof()) {
    std::getline(dump, msg);
    trim(msg);
    boost::split(tok, msg, is_any_of(", \t="), token_compress_on);

    if (startScaling == false && tok.size() == 1 && (boost::iequals(tok[0],"&END") || boost::iequals(tok[0], "/"))) {
      startScaling = true;
      index += 1;
      break;
    }
    else if(startScaling == false) {
      if (boost::iequals(tok[0].substr(0,4),"&FCI")) {
	if (boost::iequals(tok[1].substr(0,4), "NORB"))
	  norbs = atoi(tok[2].c_str());
	
	if (boost::iequals(tok[3].substr(0,5), "NELEC"))
	  nelec = atoi(tok[4].c_str());
      }
      else if (boost::iequals(tok[0].substr(0,4),"ISYM"))
	continue;
      else if (boost::iequals(tok[0].substr(0,6),"ORBSYM")) {
	for (int i=1;i<tok.size(); i++)
	  irrep.push_back(atoi(tok[i].c_str()));
      }
      else {
	for (int i=0;i<tok.size(); i++)
	  irrep.push_back(atoi(tok[i].c_str()));
      }

      index += 1;
    }
  }

  if (norbs == -1 || nelec == -1) {
    std::cout << "could not read the norbs or nelec"<<std::endl;
    exit(0);
  }
  irrep.resize(norbs);

  long npair = norbs*(norbs+1)/2;
  I2.store.resize( npair*(npair+1)/2);
  I1.store.resize(npair);
  coreE = 0.0;

  while(!dump.eof()) {
    std::getline(dump, msg);
    trim(msg);
    boost::split(tok, msg, is_any_of(", \t"), token_compress_on);
    if (tok.size() != 5)
      continue;

    double integral = atof(tok[0].c_str());int a=atoi(tok[1].c_str()), b=atoi(tok[2].c_str()), 
    c=atoi(tok[3].c_str()), d=atoi(tok[4].c_str());

    if(a==b&&b==c&&c==d&&d==0)
      coreE = integral;
    else if (c==d&&d==0)
      I1(2*(a-1),2*(b-1)) = integral;
    else
      I2(2*(a-1),2*(b-1),2*(c-1),2*(d-1)) = integral;
  }
  I2.maxEntry = *std::max_element(&I2.store[0], &I2.store[0]+I2.store.size(),myfn);
  I2.Direct = MatrixXd(norbs, norbs); I2.Direct *= 0.;
  I2.Exchange = MatrixXd(norbs, norbs); I2.Exchange *= 0.;

  for (int i=0; i<norbs; i++)
    for (int j=0; j<norbs; j++) {
      I2.Direct(i,j) = I2(2*i,2*i,2*j,2*j);
      I2.Exchange(i,j) = I2(2*i,2*j,2*i,2*j);
    }

  return;
  
}


