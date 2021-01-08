#ifndef semiStoch_HEADER_H
#define semiStoch_HEADER_H

#include <vector>
#include "Determinants.h"
#include "walkersFCIQMC.h"

class semiStoch {

 public:
  // The number of determinants in the core space
  int nDets;
  // The number of determinants in the core space on this process
  int nDetsThisProc;
  // The list of determinants in the core space
  vector<simpleDet> dets;
  // The list of determinants in the core space on this process
  vector<simpleDet> detsThisProc;
  // Holds the CI coefficients read in from the selected CI calculation
  vector<double> sciAmps;
  // Used to hold the walker amplitudes of the core determinants
  double** amps;
  // Used to hold the output of the Hamiltonian multiplied by the
  // core walker amplitudes (in the amps array)
  double** ampsFull;
  // Deterministic flags of the walkers in the main list
  vector<int> flags;
  // Hash table to find the position of a determinant in dets
  unordered_map<Determinant, int> ht;
  // The positions of non-zero elements in the core Hamiltonian.
  // pos[i,j] is the j'th non-zero column index in row i.
  vector<vector<int>> pos;
  // The values of the non-zero elements in the core Hamiltonian.
  // ham[i,j] is the j'th non-zero element in row i.
  vector<vector<int>> ham;

  semiStoch() {}

  semiStoch(std::string SHCIFile, int DetLenMin)
  {
    nDets = 0;
    nDetsThisProc = 0;

    ifstream dump(SHCIFile.c_str());

    int index = 0;
    double bestCoeff = 0.0;

    int orbsToLoopOver;
    int offset;

    orbsToLoopOver = Determinant::norbs;

    // Loop through all lines in the SHCI file
    while (dump.good())
    {
      std::string Line;
      std::getline(dump, Line);

      boost::trim_if(Line, boost::is_any_of(", \t\n"));
      
      vector<string> tok;
      boost::split(tok, Line, boost::is_any_of(", \t\n"), boost::token_compress_on);

      if (tok.size() > 2 )
      {
        double ci = atof(tok[0].c_str());
        Determinant det ;

        for (int i=0; i<orbsToLoopOver; i++)
        {
          if (boost::iequals(tok[1+i], "2")) 
          {
            det.setoccA(i, true);
            det.setoccB(i, true);
          }
          else if (boost::iequals(tok[1+i], "a")) 
          {
            det.setoccA(i, true);
            det.setoccB(i, false);
          }
          if (boost::iequals(tok[1+i], "b")) 
          {
            det.setoccA(i, false);
            det.setoccB(i, true);
          }
          if (boost::iequals(tok[1+i], "0")) 
          {
            det.setoccA(i, false);
            det.setoccB(i, false);
          }
        }

        nDets += 1;

        int proc = getProc(det, DetLenMin);
        // If the determinant belongs to this process, store it
        if (proc == commrank) {
          nDetsThisProc += 1;
          detsThisProc.push_back(det.getSimpleDet());
          sciAmps.push_back(ci);
        }
        
      }
    }
    // Finish looping over the SHCI file

    // Next we need to accumualte the core determinants from each
    // process, in the correct order (proc 0 dets, proc 1 dets, etc.)

    int determSizes[commsize], determDispls[commsize];
    int determSizesDets[commsize], determDisplsDets[commsize];

    MPI_Allgather(&nDetsThisProc, 1, MPI_INTEGER, &determSizes, 1, MPI_INTEGER, MPI_COMM_WORLD);

    determDispls[0] = 0;
    for (int i = 1; i<commsize; i++) {
      determDispls[i] = determDispls[i-1] + determSizes[i-1];
    }

    // Dets have width of 2*DetLen
    // They are stored contiguously in the vector, as required for MPI
    for (int proc=0; proc<commsize; proc++) {
      determSizesDets[proc] = determSizes[proc] * 2*DetLen;
      determDisplsDets[proc] = determDispls[proc] * 2*DetLen;
    }

    // Gather the determinants into the dets array
    dets.resize(nDets);
    MPI_Allgatherv(&detsThisProc.front(), nDetsThisProc*2*DetLen, MPI_LONG,
                   &dets.front(), determSizesDets, determDisplsDets, MPI_LONG, MPI_COMM_WORLD);

    //if (commrank == 0) {
    //  for (int i=0; i<nDets; i++) {
    //    cout << Determinant(dets[i]) << endl << flush;
    //  }
    //}
  }

};

#endif
