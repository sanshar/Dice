##########################
# User Specific Settings #
##########################

CXX = icpc

EIGEN=/home/sash2458/newApps/eigen/
#MKLLIB = /curc/sw/intel/16.0.3/mkl/lib/intel64/

FLAGS  = -O3  -std=c++17 -g  -qopenmp -I${EIGEN} #-I/curc/sw/intel/16.0.3/mkl/include/

OBJ = test.o primitives.o Integral2c.o workArray.o Integral3c.o coulomb_14_8_8.o coulomb_14_14_8.o tensor.o

LIBOBJ = primitives.o Integral2c.o workArray.o Integral3c.o coulomb_14_8_8.o coulomb_14_14_8.o tensor.o interface.o

%.o: %.cpp
	$(CXX) $(FLAGS) $(OPT) -fPIC -c $< -o $@

all: a.out libseparated.so

a.out: $(OBJ)
	$(CXX) $(FLAGS)  $(OBJ) -o a.out #-L${MKLLIB} -lmkl_gf_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt

libseparated.so: $(LIBOBJ)
	$(CXX) $(FLAGS)  $(LIBOBJ) -shared -o libseparated.so 

clean :
	find . -name "*.o"|xargs rm 2>/dev/null
