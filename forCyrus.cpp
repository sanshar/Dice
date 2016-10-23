#include "Determinants.h"
#include "CIPSIbasics.h"
#include "integral.h"
#include <vector>
#include "math.h"
#include "Hmult.h"
#include <tuple>
#include <map>
#include "Davidson.h"
#include "boost/format.hpp"
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"

using namespace std;
using namespace Eigen;
using namespace boost;

void main(int argc, char* argv[]) {

  char file [5000];
  sprintf (file, "%d-variational.bkp" , 0 );
  std::ifstream ifs(file, std::ios::binary);
  boost::archive::binary_iarchive load(ifs);
  
  int iter; std::vector<Determinants> Dets;
  load >> iter >> Dets ;
  ifs.close();

  std::vector<int> ExcitationBins(50,0);
  for (int i=1; i<Dets.size(); i++) {
    int d = Dets[0].ExcitationDistance(Dets[i]);
    ExcitationBins[d]++;
  }

}
