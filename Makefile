USE_MPI = yes
USE_INTEL = yes
#EIGEN=/projects/sash2458/apps/eigen/
#BOOST=/projects/sash2458/apps/boost_1_57_0/
#LIBIGL=/projects/sash2458/apps/libigl/include/
#EIGEN=/projects/anma2640/eigen-eigen-5a0156e40feb
#BOOST=/projects/anma2640/boost_1_66_0
#LIBIGL=/projects/anma2640/local/lib/libigl/include/
EIGEN=/projects/ilsa8974/apps/eigen/
BOOST=/projects/ilsa8974/apps/boost_1_66_0/
LIBIGL=/projects/ilsa8974/apps/libigl/include/

FLAGS = -std=c++14 -g  -O3  -I./utils -I./Wavefunctions -I${EIGEN} -I${BOOST} -I${BOOST}/include -I${LIBIGL} -I/opt/local/include/openmpi-mp/ #-DComplex
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

OBJ_VMC = obj/staticVariables.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Slater.o \
	obj/CPSSlater.o \
	obj/HFWalker.o \
	obj/AGP.o \
	obj/CPSAGP.o \
	obj/AGPWalker.o \
	obj/Pfaffian.o \
	obj/CPSPfaffian.o \
	obj/PfaffianWalker.o \
	obj/CPS.o \
	obj/excitationOperators.o\
	obj/Correlator.o \
	obj/HFWalkerHelper.o\
	obj/ShermanMorrisonWoodbury.o\
	obj/statistics.o\
	obj/evaluateE.o 


OBJ_GFMC = obj/staticVariables.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Slater.o \
	obj/CPSSlater.o \
	obj/HFWalker.o \
	obj/AGP.o \
	obj/CPSAGP.o \
	obj/AGPWalker.o \
	obj/evaluateE.o \
	obj/CPS.o \
	obj/excitationOperators.o\
	obj/HFWalkerHelper.o\
	obj/ShermanMorrisonWoodbury.o\
	obj/statistics.o\
	obj/Correlator.o


obj/%.o: %.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: Wavefunctions/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: utils/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: VMC/%.cpp  
	$(CXX) $(FLAGS) -I./VMC $(OPT) -c $< -o $@


all: bin/VMC bin/GFMC #bin/sPT  bin/GFMC

bin/GFMC	: $(OBJ_GFMC) executables/GFMC.cpp
	$(CXX)   $(FLAGS) -I./GFMC $(OPT) -c executables/GFMC.cpp -o obj/GFMC.o
	$(CXX)   $(FLAGS) $(OPT) -o  bin/GFMC $(OBJ_GFMC) obj/GFMC.o $(LFLAGS)

bin/VMC	: $(OBJ_VMC) executables/VMC.cpp
	$(CXX)   $(FLAGS) -I./VMC $(OPT) -c executables/VMC.cpp -o obj/VMC.o
	$(CXX)   $(FLAGS) $(OPT) -o  bin/VMC $(OBJ_VMC) obj/VMC.o $(LFLAGS)


bin/sPT	: $(OBJ_sPT) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/sPT $(OBJ_sPT) $(LFLAGS)

bin/CI	: $(OBJ_CI) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/CI $(OBJ_CI) $(LFLAGS)

VMC2	: $(OBJ_VMC) 
	$(CXX)   $(FLAGS) $(OPT) -o  VMC2 $(OBJ_VMC) $(LFLAGS)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm -f bin/* >/dev/null 2>&1

