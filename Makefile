CXX = mpiicpc
CC = mpiicpc
FLAGS = -std=c++11 -qopenmp -g -O2  -I/home/sash2458/apps/eigen -I/home/sash2458/apps/boost_1_57_0/ #-DComplex
DFLAGS = -std=c++11 -qopenmp -g -O2 -I/home/sash2458/apps/eigen -I/home/sash2458/apps/boost_1_57_0/ -DComplex
#FLAGS = -std=c++11 -fopenmp -w -g -O2 -I/home/james/Documents/Apps/eigen -I/home/james/Documents/Apps/boost_1_57_0/ #-DComplex
#FLAGS = -std=c++11 -qopenmp -g -O2 -I/home/sash2458/apps/eigen -I./

LFLAGS = -L/home/sash2458/apps/boost_1_57_0/stage_bla/lib -lboost_serialization -lboost_mpi -lrt
#LFLAGS = -L/home/james/Documents/Apps/boost_1_57_0/ -lboost_serialization -lboost_mpi -lrt
#LFLAGS = -L/home/sash2458/apps/boost_1_57_0/stage_janus/lib -lboost_serialization -lboost_mpi
#LFLAGS = -L/home/sash2458/apps/boost_1_57_0/stage_has/lib -lboost_serialization -lboost_mpi
#LFLAGS = -L./boost/lib -lboost_serialization -lboost_mpi

SRC_qdptsoc = QDPTSOC.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp new_anglib.cpp
SRC_hci = HCI.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp 
SRC_forcyrus = forCyrus.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp
SRC_Excitations = Excitations.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp


OBJ_qdptsoc+=obj_z/QDPTSOC.o obj_z/HCIbasics.o obj_z/Determinants.o obj_z/integral.o obj_z/input.o obj_z/Davidson.o obj_z/new_anglib.o
OBJ_hci+=obj/HCI.o obj/HCIbasics.o obj/Determinants.o obj/integral.o obj/input.o obj/Davidson.o 
OBJ_forcyrus+=obj/forCyrus.o obj/HCIbasics.o obj/Determinants.o obj/integral.o obj/input.o obj/Davidson.o
OBJ_Excitations+=obj/Excitations.o obj/HCIbasics.o obj/Determinants.o obj/integral.o obj/input.o

obj/%.o: %.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj_z/%.o: %.cpp  
	$(CXX) $(DFLAGS) $(OPT) -c $< -o $@


all: HCI stats forcyrus QDPTSOC

stats: stats.o
	$(CXX) -O3 stats.cpp -o stats
HCI	: $(OBJ_hci) 
	$(CXX)   $(FLAGS) $(OPT) -o  HCI $(OBJ_hci) $(LFLAGS)
forcyrus	: $(OBJ_forcyrus) 
	$(CXX)   $(FLAGS) $(OPT) -o  forcyrus $(OBJ_forcyrus) $(LFLAGS)
Excitations	: $(OBJ_Excitations)
	$(CXX)   $(FLAGS) $(OPT) -o  Excitations $(OBJ_Excitations) $(LFLAGS)
HCI2	: $(OBJ_hci2)
	$(CXX)   $(FLAGS) $(OPT) -o  HCI2 $(OBJ_hci2) $(LFLAGS)
QDPTSOC	: $(OBJ_qdptsoc)
	$(CXX)   $(DFLAGS) $(OPT) -o  QDPTSOC $(OBJ_qdptsoc) $(LFLAGS)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm CIST HCI 

