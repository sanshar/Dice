CXX = mpiicpc
CC = mpiicpc
FLAGS = -std=c++0x -openmp -g -O2 -I/home/sharma/apps/forServer/boost_1_53_0_mt/boost_1_53_0/  -I/home/sharma/apps/forServer/eigen
#FLAGS = -std=c++0x -g -fopenmp -I/home/sharma/apps/forServer/boost_1_53_0_mt/boost_1_53_0/  -I/home/sharma/apps/forServer/eigen
LFLAGS = -L/home/sharma/apps/forServer/boost_1_53_0_mt/boost_1_53_0/stage/lib -lboost_serialization -lboost_mpi

SRC_cisd = CISD.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp
SRC_hci = HCI.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp HCInonessentials.cpp
SRC_hci2 = HCI.cpp HCIbasics2.cpp Determinants.cpp integral.cpp input.cpp Davidson.cpp
SRC_Excitations = Excitations.cpp HCIbasics.cpp Determinants.cpp integral.cpp input.cpp

OBJ_cisd+=$(SRC_cisd:.cpp=.o)
OBJ_hci+=$(SRC_hci:.cpp=.o)
OBJ_Excitations+=$(SRC_Excitations:.cpp=.o)
OBJ_hci2+=$(SRC_hci2:.cpp=.o)

.cpp.o :
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@


all: HCI stats 

stats: stats.o
	$(CXX) -O3 stats.cpp -o stats
HCI	: $(OBJ_hci)
	$(CXX)   $(FLAGS) $(OPT) -o  HCI $(OBJ_hci) $(LFLAGS)
Excitations	: $(OBJ_Excitations)
	$(CXX)   $(FLAGS) $(OPT) -o  Excitations $(OBJ_Excitations) $(LFLAGS)
HCI2	: $(OBJ_hci2)
	$(CXX)   $(FLAGS) $(OPT) -o  HCI2 $(OBJ_hci2) $(LFLAGS)
CISD	: $(OBJ_cisd)
	$(CXX)   $(FLAGS) $(OPT) -o  CISD $(OBJ_cisd) $(LFLAGS)

clean :
	rm *.o CISD HCI
