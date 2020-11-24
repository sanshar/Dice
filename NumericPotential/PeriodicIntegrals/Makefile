##########################
# User Specific Settings #
##########################

CXX = icpc
CC = icc

EIGEN=/projects/sash2458/newApps/eigen/
BOOST=/projects/sash2458/newApps/boost_1_67_0/
#EIGEN=/usr/include/eigen3
#BOOST=/usr/include/boost
#MKLLIB = /curc/sw/intel/16.0.3/mkl/lib/intel64/

FLAGS  = -DNDEBUG -O3  -std=c++17 -g  -fopenmp -I${EIGEN} -I${BOOST} #-I/curc/sw/intel/16.0.3/mkl/include/
#FLAGS  =  -std=c++17 -g  -fopenmp -I${EIGEN} -I${BOOST} #-I/curc/sw/intel/16.0.3/mkl/include/

OBJ = main.o BasisShell.o interface.o GeneratePolynomials.o CxMemoryStack.o IrAmrr.o Integral2c_Boys.o IrBoysFn.o Kernel.o LatticeSum.o

LIBOBJ = BasisShell.o interface.o GeneratePolynomials.o

%.o: %.cpp
	$(CXX) $(FLAGS) $(OPT) -fPIC -c $< -o $@

%.o: %.c
	$(CC) -O3 -g -qopenmp -fPIC -c $< -o $@

all: a.out libseparated.so

a.out: $(OBJ)
	$(CXX) $(FLAGS)  $(OBJ) -o a.out #-L${MKLLIB} -lmkl_gf_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt

libseparated.so: $(LIBOBJ)
	$(CXX) $(FLAGS)  $(LIBOBJ) -shared -o libseparated.so 

clean :
	find . -name "*.o"|xargs rm 2>/dev/null
