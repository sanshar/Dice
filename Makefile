USE_MPI = yes
USE_INTEL = yes
EIGEN=/home/anma2640/projects/eigen-eigen-5a0156e40feb
BOOST=/home/anma2640/projects/boost_1_66_0
#EIGEN=/home/mussard/softwares/eigen
#BOOST=/home/mussard/softwares/boost_1_64_0

#FLAGS = -std=c++11 -g  -I${EIGEN} -I${BOOST} #-DComplex
FLAGS = -std=c++11 -g  -O3  -I${EIGEN} -I${BOOST} #-DComplex
DFLAGS = -std=c++11 -g -O3 -I${EIGEN} -I${BOOST} -DComplex -DGreen 

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
		LFLAGS = -L${BOOST}/stage_bla/lib -lboost_serialization -lrt
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
		LFLAGS = -L${BOOST}/stage_bla/lib -lboost_serialization -lrt
		FLAGS += -DSERIAL
		DFLAGS += -DSERIAL
	endif
endif

SRC_qdptsoc = QDPTSOC.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp new_anglib.cpp SOChelper.cpp SHCIgetdeterminants.cpp SHCIsampledeterminants.cpp SHCIrdm.cpp SHCISortMpiUtils.cpp SHCImakeHamiltonian.cpp
SRC_GTensorFT = GTensorFT.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp new_anglib.cpp SOChelper.cpp SHCIgetdeterminants.cpp SHCIsampledeterminants.cpp SHCIrdm.cpp SHCISortMpiUtils.cpp SHCImakeHamiltonian.cpp
SRC_GTensorFT2 = GTensorFT2.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp new_anglib.cpp SOChelper.cpp SHCIgetdeterminants.cpp SHCIsampledeterminants.cpp SHCIrdm.cpp SHCISortMpiUtils.cpp SHCImakeHamiltonian.cpp
SRC_Dice = SHCI.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp SHCIgetdeterminants.cpp SHCIsampledeterminants.cpp SHCIrdm.cpp SHCISortMpiUtils.cpp SHCImakeHamiltonian.cpp SHCIshm.cpp
SRC_ZDice = SHCI.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp SOChelper.cpp new_anglib.cpp SHCIgetdeterminants.cpp SHCIsampledeterminants.cpp SHCIrdm.cpp SHCISortMpiUtils.cpp SHCImakeHamiltonian.cpp SHCIshm.cpp
SRC_forcyrus = forCyrus.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp
SRC_Excitations = Excitations.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp
SRC_GreensFunction = GreensFunction.cpp Determinants.cpp integral.cpp Davidson.cpp SHCISortMpiUtils.cpp SHCImakeHamiltonian.cpp SHCIshm.cpp

OBJ_qdptsoc+=obj_z/QDPTSOC.o obj_z/SHCIbasics.o obj_z/Determinants.o obj_z/integral.o obj_z/input.o obj_z/Davidson.o obj_z/new_anglib.o obj_z/SOChelper.o obj_z/SHCIgetdeterminants.o obj_z/SHCIsampledeterminants.o obj_z/SHCIrdm.o obj_z/SHCISortMpiUtils.o obj_z/SHCImakeHamiltonian.o
OBJ_gtensorft+=obj_z/GTensorFT.o obj_z/SHCIbasics.o obj_z/Determinants.o obj_z/integral.o obj_z/input.o obj_z/Davidson.o obj_z/new_anglib.o obj_z/SOChelper.o obj_z/SHCIgetdeterminants.o obj_z/SHCIsampledeterminants.o obj_z/SHCIrdm.o obj_z/SHCISortMpiUtils.o obj_z/SHCImakeHamiltonian.o
OBJ_gtensorft2+=obj_z/GTensorFT2.o obj_z/SHCIbasics.o obj_z/Determinants.o obj_z/integral.o obj_z/input.o obj_z/Davidson.o obj_z/new_anglib.o obj_z/SOChelper.o obj_z/SHCIgetdeterminants.o obj_z/SHCIsampledeterminants.o obj_z/SHCIrdm.o obj_z/SHCISortMpiUtils.o obj_z/SHCImakeHamiltonian.o
OBJ_Dice+=obj/SHCI.o obj/SHCIbasics.o obj/Determinants.o obj/integral.o obj/input.o obj/Davidson.o obj/SHCIgetdeterminants.o  obj/SHCIsampledeterminants.o obj/SHCIrdm.o obj/SHCISortMpiUtils.o obj/SHCImakeHamiltonian.o obj/SHCIshm.o
OBJ_ZDice+=obj_z/SHCI.o obj_z/SHCIbasics.o obj_z/Determinants.o obj_z/integral.o obj_z/input.o obj_z/Davidson.o obj_z/SOChelper.o obj_z/new_anglib.o obj_z/SHCIgetdeterminants.o  obj_z/SHCIsampledeterminants.o obj_z/SHCIrdm.o obj_z/SHCISortMpiUtils.o obj_z/SHCImakeHamiltonian.o
OBJ_forcyrus+=obj/forCyrus.o obj/SHCIbasics.o obj/Determinants.o obj/integral.o obj/input.o obj/Davidson.o obj/SHCIgetdeterminants.o  obj/SHCIsampledeterminants.o obj/SHCIrdm.o obj/SHCISortMpiUtils.o obj/SHCImakeHamiltonian.o
OBJ_Excitations+=obj/Excitations.o obj/SHCIbasics.o obj/Determinants.o obj/integral.o obj/input.o
OBJ_GreensFunction+=obj_z/GreensFunction.o obj_z/Determinants.o obj_z/integral.o obj_z/Davidson.o obj_z/SHCISortMpiUtils.o obj_z/SHCImakeHamiltonian.o obj_z/SHCIshm.o

obj/%.o: %.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj_z/%.o: %.cpp  
	$(CXX) $(DFLAGS) $(OPT) -c $< -o $@


all: Dice stats QDPTSOC ZDice #GTensorFT GTensorFT2

stats: stats.o
	$(CXX) -O3 stats.cpp -o stats
Dice	: $(OBJ_Dice) 
	$(CXX)   $(FLAGS) $(OPT) -o  Dice $(OBJ_Dice) $(LFLAGS)
ZDice	: $(OBJ_ZDice) 
	$(CXX)   $(FLAGS) $(OPT) -o  ZDice $(OBJ_ZDice) $(LFLAGS)
forcyrus	: $(OBJ_forcyrus) 
	$(CXX)   $(FLAGS) $(OPT) -o  forcyrus $(OBJ_forcyrus) $(LFLAGS)
Excitations	: $(OBJ_Excitations)
	$(CXX)   $(FLAGS) $(OPT) -o  Excitations $(OBJ_Excitations) $(LFLAGS)
SHCI2	: $(OBJ_shci2)
	$(CXX)   $(FLAGS) $(OPT) -o  SHCI2 $(OBJ_shci2) $(LFLAGS)
QDPTSOC	: $(OBJ_qdptsoc)
	$(CXX)   $(DFLAGS) $(OPT) -o  QDPTSOC $(OBJ_qdptsoc) $(LFLAGS)
GTensorFT	: $(OBJ_gtensorft)
	$(CXX)   $(DFLAGS) $(OPT) -o  GTensorFT $(OBJ_gtensorft) $(LFLAGS)
GTensorFT2	: $(OBJ_gtensorft2)
	$(CXX)   $(DFLAGS) $(OPT) -o  GTensorFT2 $(OBJ_gtensorft2) $(LFLAGS)
GreensFunction : $(OBJ_GreensFunction) 
	$(CXX)   $(DFLAGS) $(OPT) -o  GreensFunction $(OBJ_GreensFunction) $(LFLAGS)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm CIST Dice ZDice QDPTSOC GTensorFT forcyrus >/dev/null 2>&1

