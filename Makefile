USE_MPI = no
USE_INTEL = no
EIGEN=/Users/sandeepsharma/Academics/Programs/Eigen/
BOOST=/opt/local/include
LIBIGL=/Users/sandeepsharma/Academics/Programs/libigl/include/
#EIGEN=/projects/anma2640/eigen-eigen-5a0156e40feb
#BOOST=/projects/anma2640/boost_1_66_0

FLAGS = -std=c++11 -g  -O3  -I./ -I./utils -I./optimizer/ -I./Wavefunctions -I${EIGEN} -I${BOOST} -I${LIBIGL} -I/opt/local/include/openmpi-mp/ #-DComplex
#FLAGS = -std=c++11  -g  -I./ -I./utils -I./optimizer/ -I./Wavefunctions -I${EIGEN} -I${BOOST} -I${LIBIGL} -I/opt/local/include/openmpi-mp/ #-DComplex




ifeq ($(USE_INTEL), yes) 
	FLAGS += -qopenmp
	DFLAGS += -qopenmp
	ifeq ($(USE_MPI), yes) 
		CXX = mpiicpc
		CC = mpiicpc
		LFLAGS = -L${BOOST}/stage/lib -lboost_serialization -lboost_mpi
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
		LFLAGS = -L/opt/local/lib -lboost_serialization-mt -lboost_mpi-mt
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

SRC_VMC = VMC.cpp MoDeterminants.cpp staticVariables.cpp input.cpp integral.cpp SHCIshm.cpp CPS.cpp Wfn.cpp evaluateE.cpp Determinants.cpp diis.cpp Walker.cpp evaluatePT.cpp

SRC_sPT = StochasticPT.cpp staticVariables.cpp input.cpp integral.cpp SHCIshm.cpp CPS.cpp Wavefunctions/CPSSlater.cpp evaluateE.cpp Determinants.cpp  Walker.cpp  evaluatePT.cpp

OBJ_PythonInterface = obj/PythonInterface.o \
		obj/staticVariables.o \
		obj/input.o \
		obj/integral.o\
		obj/SHCIshm.o \
		obj/Determinants.o \
		obj/CPSSlater.o \
		obj/HFWalker.o \
		obj/CPS.o \
		obj/evaluateE.o \
		obj/Davidson.o 

SRC_CI = ConfigurationInteraction.cpp MoDeterminants.cpp staticVariables.cpp input.cpp integral.cpp SHCIshm.cpp CPS.cpp Wfn.cpp evaluateE.cpp Determinants.cpp diis.cpp Walker.cpp Davidson.cpp evaluatePT.cpp

SRC_GFMC = GFMC.cpp MoDeterminants.cpp staticVariables.cpp input.cpp integral.cpp SHCIshm.cpp CPS.cpp Wfn.cpp evaluateE.cpp Determinants.cpp diis.cpp Walker.cpp optimizer.cpp Davidson.cpp evaluatePT.cpp

OBJ_VMC+=obj/VMC.o obj/MoDeterminants.o obj/staticVariables.o obj/input.o obj/integral.o obj/SHCIshm.o obj/CPS.o obj/Wfn.o obj/evaluateE.o obj/Determinants.o obj/diis.o obj/Walker.o obj/optimizer.o obj/Davidson.o obj/evaluatePT.o

OBJ_sPT+=obj/StochasticPT.o obj/MoDeterminants.o obj/staticVariables.o obj/input.o obj/integral.o obj/SHCIshm.o obj/CPS.o obj/Wfn.o obj/evaluateE.o obj/Determinants.o obj/Walker.o obj/evaluatePT.o

OBJ_CI+=obj/ConfigurationInteraction.o obj/MoDeterminants.o obj/staticVariables.o obj/input.o obj/integral.o obj/SHCIshm.o obj/CPS.o obj/Wfn.o obj/evaluateE.o obj/Determinants.o obj/Walker.o obj/evaluatePT.o obj/Davidson.o

#OBJ_PythonInterface+=obj/PythonInterface.o obj/staticVariables.o obj/input.o obj/integral.o obj/SHCIshm.o obj/CPS.o obj/CPSSlater.o obj/CPSSlaterWalker.o obj/evaluateE.o obj/Determinants.o obj/Davidson.o

OBJ_GFMC+=obj/GFMC.o obj/MoDeterminants.o obj/staticVariables.o obj/input.o obj/integral.o obj/SHCIshm.o obj/CPS.o obj/Wfn.o obj/evaluateE.o obj/Determinants.o obj/Walker.o obj/Davidson.o


obj/%.o: %.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: Wavefunctions/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: utils/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: optimizer/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: executables/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@


all: bin/PythonInterface bin/sPT  bin/GFMC bin/CI

bin/GFMC	: $(OBJ_GFMC) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/GFMC $(OBJ_GFMC) $(LFLAGS)

bin/PythonInterface	: $(OBJ_PythonInterface) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/PythonInterface $(OBJ_PythonInterface) $(LFLAGS)


VMC	: $(OBJ_VMC) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/VMC $(OBJ_VMC) $(LFLAGS)

bin/sPT	: $(OBJ_sPT) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/sPT $(OBJ_sPT) $(LFLAGS)

bin/CI	: $(OBJ_CI) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/CI $(OBJ_CI) $(LFLAGS)

VMC2	: $(OBJ_VMC) 
	$(CXX)   $(FLAGS) $(OPT) -o  VMC2 $(OBJ_VMC) $(LFLAGS)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm -f bin/* >/dev/null 2>&1

