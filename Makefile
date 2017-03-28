CXX = mpiicpc
CC = mpiicpc
FLAGS = -std=c++11 -qopenmp -g -O2 -I/home/sash2458/apps/eigen -I/home/sash2458/apps/boost_1_57_0/ #-DComplex
DFLAGS = -std=c++11 -qopenmp -g -O2 -I/home/sash2458/apps/eigen -I/home/sash2458/apps/boost_1_57_0/ -DComplex
#FLAGS = -std=c++11 -fopenmp -w -g -O2 -I/home/james/Documents/Apps/eigen -I/home/james/Documents/Apps/boost_1_57_0/ #-DComplex
#FLAGS = -std=c++11 -qopenmp -g -O2 -I/home/sash2458/apps/eigen -I./

LFLAGS = -L/home/sash2458/apps/boost_1_57_0/stage_bla/lib -lboost_serialization -lboost_mpi -lrt
#LFLAGS = -L/home/james/Documents/Apps/boost_1_57_0/ -lboost_serialization -lboost_mpi -lrt
#LFLAGS = -L/home/sash2458/apps/boost_1_57_0/stage_janus/lib -lboost_serialization -lboost_mpi
#LFLAGS = -L/home/sash2458/apps/boost_1_57_0/stage_has/lib -lboost_serialization -lboost_mpi
#LFLAGS = -L./boost/lib -lboost_serialization -lboost_mpi

SRC_qdptsoc = QDPTSOC.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp new_anglib.cpp SOChelper.cpp SHCIgetdeterminants.cpp SHCIsampledeterminants.cpp
SRC_shci = SHCI.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp SHCIgetdeterminants.cpp SHCIsampledeterminants.cpp
SRC_zshci = SHCI.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp SOChelper.cpp new_anglib.cpp SHCIgetdeterminants.cpp SHCIsampledeterminants.cpp
SRC_forcyrus = forCyrus.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp
SRC_Excitations = Excitations.cpp SHCIbasics.cpp Determinants.cpp integral.cpp input.cpp


OBJ_qdptsoc+=obj_z/QDPTSOC.o obj_z/SHCIbasics.o obj_z/Determinants.o obj_z/integral.o obj_z/input.o obj_z/Davidson.o obj_z/new_anglib.o obj_z/SOChelper.o obj_z/SHCIgetdeterminants.o obj_z/SHCIsampledeterminants.o
OBJ_shci+=obj/SHCI.o obj/SHCIbasics.o obj/Determinants.o obj/integral.o obj/input.o obj/Davidson.o obj/SHCIgetdeterminants.o  obj/SHCIsampledeterminants.o
OBJ_zshci+=obj_z/SHCI.o obj_z/SHCIbasics.o obj_z/Determinants.o obj_z/integral.o obj_z/input.o obj_z/Davidson.o obj_z/SOChelper.o obj_z/new_anglib.o obj_z/SHCIgetdeterminants.o  obj_z/SHCIsampledeterminants.o
OBJ_forcyrus+=obj/forCyrus.o obj/SHCIbasics.o obj/Determinants.o obj/integral.o obj/input.o obj/Davidson.o obj/SHCIgetdeterminants.o  obj/SHCIsampledeterminants.o
OBJ_Excitations+=obj/Excitations.o obj/SHCIbasics.o obj/Determinants.o obj/integral.o obj/input.o

obj/%.o: %.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj_z/%.o: %.cpp  
	$(CXX) $(DFLAGS) $(OPT) -c $< -o $@


all: SHCI stats forcyrus QDPTSOC ZSHCI

stats: stats.o
	$(CXX) -O3 stats.cpp -o stats
SHCI	: $(OBJ_shci) 
	$(CXX)   $(FLAGS) $(OPT) -o  SHCI $(OBJ_shci) $(LFLAGS)
ZSHCI	: $(OBJ_zshci) 
	$(CXX)   $(FLAGS) $(OPT) -o  ZSHCI $(OBJ_zshci) $(LFLAGS)
forcyrus	: $(OBJ_forcyrus) 
	$(CXX)   $(FLAGS) $(OPT) -o  forcyrus $(OBJ_forcyrus) $(LFLAGS)
Excitations	: $(OBJ_Excitations)
	$(CXX)   $(FLAGS) $(OPT) -o  Excitations $(OBJ_Excitations) $(LFLAGS)
SHCI2	: $(OBJ_shci2)
	$(CXX)   $(FLAGS) $(OPT) -o  SHCI2 $(OBJ_shci2) $(LFLAGS)
QDPTSOC	: $(OBJ_qdptsoc)
	$(CXX)   $(DFLAGS) $(OPT) -o  QDPTSOC $(OBJ_qdptsoc) $(LFLAGS)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm CIST SHCI 

