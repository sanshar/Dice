#include "global.h"
#include "OccRestrictions.h"
#include "input.h"

void OccRestrictions::setElec(vector<int>& closed) {
  currentElec = 0;
  for (int i=0; i<closed.size(); i++)
    for (int x=0; x<orbs.size(); x++)
      if (closed[i] == orbs[x]) currentElec++;
}
  
void OccRestrictions::setElec(int elec){
  currentElec = elec;
}

bool OccRestrictions::oneElecAllowed(int i, int a) {
  int elec = currentElec;
  for (int x=0; x<orbs.size(); x++) {
    if (orbs[x] == i) elec--;
    if (orbs[x] == a) elec++;
  }

  if (elec < minElec || elec > maxElec) return false;
  return true;
}

bool OccRestrictions::twoElecAllowed(int i, int j, int a, int b) {
  int elec = currentElec;
  for (int x=0; x<orbs.size(); x++) {
    if (orbs[x] == i || orbs[x] == j) elec--;
    if (orbs[x] == a || orbs[x] == b) elec++;
  }

  if (elec < minElec || elec > maxElec) return false;
  return true;
}
  
void initiateRestrictions(schedule& schd, vector<int>& closed) {
  for (int i=0; i<schd.restrictions.size(); i++) {
    OccRestrictions& res = schd.restrictions[i];
    res.setElec(closed);
    
    if (res.currentElec < res.minElec ||
        res.currentElec > res.maxElec ) {
      Determinant det;
      for (int k=0; k<closed.size(); k++)
        det.setocc(closed[k], true);

      cout << "Determinant: "<<det<<endl;
      cout << "num elecs in orbs: "<<res.currentElec<<endl;
      cout << "does not satisfy the restriction "<<schd.restrictions[i]<<endl;
      cout << "most likely the occupation of the initial determinant does not satisfy the restriction"<<endl;
      exit(0);
    }
  }
}


bool satisfiesRestrictions(schedule& schd, int i, int a) {
  for (int x=0; x<schd.restrictions.size(); x++) {
    OccRestrictions& res = schd.restrictions[x];
    if (!res.oneElecAllowed(i, a)) return false;
  }
  return true;
}

bool satisfiesRestrictions(schedule& schd, int i, int j, int a, int b) {
  for (int x=0; x<schd.restrictions.size(); x++) {
    OccRestrictions& res = schd.restrictions[x];
    if (!res.twoElecAllowed(i, j, a, b)) return false;
  }
  return true;
}

