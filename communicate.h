#ifndef COMMUNICATE_HEADER_H
#define COMMUNICATE_HEADER_H

#ifndef SERIAL
  #include <boost/mpi.hpp>
  #include <boost/mpi/communicator.hpp>
  

  inline int mpigetrank() { boost::mpi::communicator world; return world.rank(); }
  inline int mpigetsize() { boost::mpi::communicator world; return world.size(); }


#else
  inline int mpigetrank() { return 0; }
  inline int mpigetsize() { return 1; }
#endif

#define pout if (mpigetrank() == 0) std::cout

#endif
