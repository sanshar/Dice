CXX = mpiicpc
CC = mpiicpc
FLAGS = -std=c++11 -qopenmp -g -O2 -I/home/sash2458/apps/eigen -I/home/sash2458/apps/boost_1_57_0/ #-DComplex
#FLAGS = -std=c++11 -fopenmp -w -g -O2 -I/home/james/Documents/Apps/eigen -I/home/james/Documents/Apps/boost_1_57_0/ #-DComplex
#FLAGS = -std=c++11 -qopenmp -g -O2 -I/home/sash2458/apps/eigen -I./

LFLAGS = -L/home/sash2458/apps/boost_1_57_0/stage_bla/lib -lboost_serialization -lboost_mpi -lrt
#LFLAGS = -L/home/james/Documents/Apps/boost_1_57_0/ -lboost_serialization -lboost_mpi -lrt
#LFLAGS = -L/home/sash2458/apps/boost_1_57_0/stage_janus/lib -lboost_serialization -lboost_mpi
#LFLAGS = -L/home/sash2458/apps/boost_1_57_0/stage_has/lib -lboost_serialization -lboost_mpi
#LFLAGS = -L./boost/lib -lboost_serialization -lboost_mpi

SRC_cisd = CISD.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp
SRC_hci = HCI.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp 
SRC_forcyrus = forCyrus.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp
SRC_hci2 = HCI.cpp HCIbasics2.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp
SRC_Excitations = Excitations.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp


OBJ_cisd+=$(SRC_cisd:.cpp=.o)
OBJ_hci+=$(SRC_hci:.cpp=.o)
OBJ_forcyrus+=$(SRC_forcyrus:.cpp=.o)
OBJ_Excitations+=$(SRC_Excitations:.cpp=.o)
OBJ_hci2+=$(SRC_hci2:.cpp=.o)

.cpp.o :
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@


all: HCI stats forcyrus

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
CISD	: $(OBJ_cisd)
	$(CXX)   $(FLAGS) $(OPT) -o  CISD $(OBJ_cisd) $(LFLAGS)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm CIST HCI 

