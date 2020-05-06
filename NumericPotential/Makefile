##########################
# User Specific Settings #
##########################

CXX = mpiicpc

PYSCF=/home/sash2458/newApps/pyscf/
EIGEN=/projects/sash2458/newApps/eigen/
PVFMM=/projects/sash2458/newApps/FMM/pvfmm/
BOOST=/home/sash2458/newApps/boost_1_67_0/
MKL=/curc/sw/intel/17.4/mkl/

FLAGS  = -O3 -qno-offload -Qoption,cpp,--extended_float_type -std=c++17 -g -fPIC -qopenmp -I$(PVFMM)/include -I${EIGEN} -I${BOOST} -I/opt/local/include/openmpi-mp/ -I${MKL}/include

OBJ = fittedFunctions.o  CxLebedevGrid.o  CalculateSphHarmonics.o  pythonInterface.o 

%.o: %.cpp
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@

all: ../bin/libPeriodic.so

../bin/libPeriodic.so: $(OBJ)
	$(CXX) $(FLAGS) -shared $(OBJ) -o libPeriodic.so -L${PYSCF}pyscf/lib/  -L${PYSCF}pyscf/lib/deps/lib -lcvhf -lcint -lcgto -L$(PVFMM) -lpvfmm -lfftw3 -ldft -L${MKL}/lib/intel64 -lmkl_rt -lpthread -lm -ldl
	cp libPeriodic.so ../bin/.
	cp libPeriodic.so /home/sash2458/newApps/pyscf/pyscf/lib

clean :
	find . -name "*.o"|xargs rm 2>/dev/null
