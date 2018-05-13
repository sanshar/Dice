USE_MPI = yes
USE_INTEL = yes
EIGEN=/projects/sash2458/apps/eigen/
BOOST=/projects/sash2458/apps/boost_1_57_0/
#EIGEN=/projects/anma2640/eigen-eigen-5a0156e40feb
#BOOST=/projects/anma2640/boost_1_66_0

FLAGS = -std=c++11 -g  -O3  -I${EIGEN} -I${BOOST} #-DComplex
#FLAGS = -std=c++11 -g   -I${EIGEN} -I${BOOST} #-DComplex

ifeq ($(USE_INTEL), yes) 
	FLAGS += -qopenmp
	DFLAGS += -qopenmp
	ifeq ($(USE_MPI), yes) 
		CXX = mpiicpc
		CC = mpiicpc
		LFLAGS = -L${BOOST}/stage/lib -lboost_serialization -lboost_mpi -lrt
	else
		CXX = icpc
		CC = icpc
		LFLAGS = -L${BOOST}/stage/lib -lboost_serialization -lrt
		FLAGS += -DSERIAL
		DFLAGS += -DSERIAL
	endif
else
	FLAGS += -fopenmp
	DFLAGS += -fopenmp
	ifeq ($(USE_MPI), yes) 
		CXX = mpicxx
		CC = mpicxx
		LFLAGS = -L/home/mahajank/lib -lboost_serialization -lboost_mpi -lrt
	else
		CXX = g++
		CC = g++
		LFLAGS = -L${BOOST}/stage/lib -lboost_serialization -lrt
		FLAGS += -DSERIAL
		DFLAGS += -DSERIAL
	endif
endif

# Host specific configurations.
HOSTNAME := $(shell hostname)
ifneq ($(filter dft node%, $(HOSTNAME)),)
include dft.mk
endif

SRC_VMC = VMC.cpp MoDeterminants.cpp staticVariables.cpp input.cpp integral.cpp SHCIshm.cpp CPS.cpp Wfn.cpp evaluateE.cpp Determinants.cpp diis.cpp Walker.cpp optimizer.cpp Davidson.cpp evaluatePT.cpp

SRC_sPT = StochasticPT.cpp MoDeterminants.cpp staticVariables.cpp input.cpp integral.cpp SHCIshm.cpp CPS.cpp Wfn.cpp evaluateE.cpp Determinants.cpp diis.cpp Walker.cpp optimizer.cpp Davidson.cpp evaluatePT.cpp

SRC_PythonInteface = PythonInterface.cpp MoDeterminants.cpp staticVariables.cpp input.cpp integral.cpp SHCIshm.cpp CPS.cpp Wfn.cpp evaluateE.cpp Determinants.cpp diis.cpp Walker.cpp optimizer.cpp Davidson.cpp evaluatePT.cpp

OBJ_VMC+=obj/VMC.o obj/MoDeterminants.o obj/staticVariables.o obj/input.o obj/integral.o obj/SHCIshm.o obj/CPS.o obj/Wfn.o obj/evaluateE.o obj/Determinants.o obj/diis.o obj/Walker.o obj/optimizer.o obj/Davidson.o obj/evaluatePT.o

OBJ_sPT+=obj/StochasticPT.o obj/MoDeterminants.o obj/staticVariables.o obj/input.o obj/integral.o obj/SHCIshm.o obj/CPS.o obj/Wfn.o obj/evaluateE.o obj/Determinants.o obj/diis.o obj/Walker.o obj/optimizer.o obj/Davidson.o obj/evaluatePT.o

OBJ_PythonInterface+=obj/PythonInterface.o obj/MoDeterminants.o obj/staticVariables.o obj/input.o obj/integral.o obj/SHCIshm.o obj/CPS.o obj/Wfn.o obj/evaluateE.o obj/Determinants.o obj/diis.o obj/Walker.o obj/optimizer.o obj/Davidson.o obj/evaluatePT.o


obj/%.o: %.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj_z/%.o: %.cpp  
	$(CXX) $(DFLAGS) $(OPT) -c $< -o $@


all: VMC PythonInterface sPT

VMC	: $(OBJ_VMC) 
	$(CXX)   $(FLAGS) $(OPT) -o  VMC $(OBJ_VMC) $(LFLAGS)

sPT	: $(OBJ_sPT) 
	$(CXX)   $(FLAGS) $(OPT) -o  sPT $(OBJ_sPT) $(LFLAGS)

PythonInterface	: $(OBJ_PythonInterface) 
	$(CXX)   $(FLAGS) $(OPT) -o  PythonInterface $(OBJ_PythonInterface) $(LFLAGS)

VMC2	: $(OBJ_VMC) 
	$(CXX)   $(FLAGS) $(OPT) -o  VMC2 $(OBJ_VMC) $(LFLAGS)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm -f VMC >/dev/null 2>&1

