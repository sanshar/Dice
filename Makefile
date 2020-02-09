USE_MPI = yes
USE_INTEL = yes
EIGEN=/projects/sash2458/newApps/eigen/
#BOOST=/projects/sash2458/apps/boost_1_57_0/
LIBIGL=/projects/sash2458/apps/libigl/include/
HDF5=/curc/sw/hdf5/1.10.1/impi/17.3/intel/17.4/
#EIGEN=/projects/ilsa8974/apps/eigen/
BOOST=/projects/ilsa8974/apps/boost_1_66_0/
#LIBIGL=/projects/ilsa8974/apps/libigl/include/
SPARSEHASH=/home/nsb37/local/sparsehash

FLAGS = -std=c++14 -g  -O3 -I./VMC -I./utils -I./Wavefunctions -I./ICPT -I./ICPT/StackArray/ -I${EIGEN} -I${BOOST} -I${BOOST}/include -I${LIBIGL} -I${HDF5}/include -I${SPARSEHASH}/include -I/opt/local/include/openmpi-mp/  #-DComplex
#FLAGS = -std=c++14 -g   -I./utils -I./Wavefunctions -I${EIGEN} -I${BOOST} -I${BOOST}/include -I${LIBIGL} -I/opt/local/include/openmpi-mp/ #-DComplex

INCLUDE_MKL=-I/curc/sw/intel/16.0.3/mkl/include
LIB_MKL = -L/curc/sw/intel/16.0.3/mkl/lib/intel64/ -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core




ifeq ($(USE_INTEL), yes) 
	FLAGS += -qopenmp
	DFLAGS += -qopenmp
	ifeq ($(USE_MPI), yes) 
		CXX = mpiicpc
		CC = mpiicpc
		LFLAGS = -L${BOOST}/stage/lib -lboost_serialization -lboost_mpi -lboost_program_options -L${HDF5}/lib -lhdf5
	else
		CXX = icpc
		CC = icpc
		LFLAGS = -L${BOOST}/stage/lib -lboost_serialization-mt
		FLAGS += -DSERIAL
		DFLAGS += -DSERIAL
	endif
else
	FLAGS += -openmp
	DFLAGS += -openmp
	ifeq ($(USE_MPI), yes) 
		CXX = mpicxx
		CC = mpicxx
		LFLAGS = -L/opt/local/lib -lboost_serialization-mt -lboost_mpi-mt -lboost_program_options-mt
	else
		CXX = g++
		CC = g++
		LFLAGS = -L/opt/local/lib -lboost_serialization-mt
		FLAGS += -DSERIAL
		DFLAGS += -DSERIAL
	endif
endif

# Host specific configurations.
HOSTNAME := $(shell hostname)
ifneq ($(filter dft node%, $(HOSTNAME)),)
include dft.mk
endif

OBJ_VMC = obj/staticVariables.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Slater.o \
	obj/AGP.o \
	obj/Pfaffian.o \
	obj/Jastrow.o \
	obj/Gutzwiller.o \
	obj/CPS.o \
	obj/RBM.o \
	obj/JRBM.o \
	obj/Correlator.o \
	obj/SelectedCI.o \
	obj/SimpleWalker.o \
	obj/ShermanMorrisonWoodbury.o\
	obj/excitationOperators.o\
    obj/statistics.o \
    obj/sr.o \
    obj/evaluateE.o 

OBJ_ICPT= obj/PerturberDependentCode.o \
	obj/BlockContract.o \
	obj/CxAlgebra.o \
	obj/CxIndentStream.o \
	obj/CxNumpyArray.o \
	obj/icpt.o \
	obj/CxMemoryStack.o \
	obj/CxStorageDevice.o \
	obj/TensorTranspose.o

OBJ_GFMC = obj/staticVariables.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Slater.o \
	obj/AGP.o \
	obj/Pfaffian.o \
	obj/Jastrow.o \
	obj/Gutzwiller.o \
	obj/CPS.o \
	obj/evaluateE.o \
	obj/excitationOperators.o\
	obj/ShermanMorrisonWoodbury.o\
	obj/statistics.o \
	obj/sr.o \
	obj/Correlator.o


OBJ_FCIQMC = obj/staticVariables.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Correlator.o \
	obj/spawnFCIQMC.o \
	obj/walkersFCIQMC.o \


obj/%.o: %.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: Wavefunctions/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: utils/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: VMC/%.cpp  
	$(CXX) $(FLAGS) -I./VMC $(OPT) -c $< -o $@
obj/%.o: FCIQMC/%.cpp  
	$(CXX) $(FLAGS) -I./FCIQMC $(OPT) -c $< -o $@
obj/%.o: ICPT/%.cpp  
	$(CXX) $(FLAGS) $(INCLUDE_MKL) -I./ICPT/TensorExpressions/ $(OPT) -c $< -o $@
obj/%.o: ICPT/StackArray/%.cpp  
	$(CXX) $(FLAGS) $(INCLUDE_MKL) $(OPT) -c $< -o $@


all: bin/VMC bin/GFMC bin/FCIQMC bin/ICPT #bin/sPT  bin/GFMC

bin/GFMC	: $(OBJ_GFMC) executables/GFMC.cpp
	$(CXX)   $(FLAGS) -I./GFMC $(OPT) -c executables/GFMC.cpp -o obj/GFMC.o
	$(CXX)   $(FLAGS) $(OPT) -o  bin/GFMC $(OBJ_GFMC) obj/GFMC.o $(LFLAGS) 

bin/ICPT	: $(OBJ_ICPT) executables/ICPT.cpp
	$(CXX)   $(FLAGS) $(INCLUDE_MKL)  $(OPT) -c executables/ICPT.cpp -o obj/ICPT.o
	$(CXX)   $(FLAGS) $(OPT) -o  bin/ICPT $(OBJ_ICPT) obj/ICPT.o $(LFLAGS) $(LIB_MKL)

bin/VMC	: $(OBJ_VMC) executables/VMC.cpp
	$(CXX)   $(FLAGS) -I./VMC $(OPT) -c executables/VMC.cpp -o obj/VMC.o
	$(CXX)   $(FLAGS) $(OPT) -o  bin/VMC $(OBJ_VMC) obj/VMC.o $(LFLAGS)

bin/FCIQMC	: $(OBJ_FCIQMC) executables/FCIQMC.cpp
	$(CXX)   $(FLAGS) -I./FCIQMC $(OPT) -c executables/FCIQMC.cpp -o obj/FCIQMC.o
	$(CXX)   $(FLAGS) $(OPT) -o  bin/FCIQMC $(OBJ_FCIQMC) obj/FCIQMC.o $(LFLAGS)

bin/sPT	: $(OBJ_sPT) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/sPT $(OBJ_sPT) $(LFLAGS)

bin/CI	: $(OBJ_CI) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/CI $(OBJ_CI) $(LFLAGS)

VMC2	: $(OBJ_VMC) 
	$(CXX)   $(FLAGS) $(OPT) -o  VMC2 $(OBJ_VMC) $(LFLAGS)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm -f bin/* >/dev/null 2>&1

