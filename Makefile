CXX = mpiicpc
CC = mpiicpc
FLAGS = -std=c++0x -fopenmp -g -O3 -I/home/sharma/apps/forServer/boost_1_53_0_mt/boost_1_53_0/  -I/home/sharma/apps/forServer/eigen
#FLAGS = -std=c++0x -g -openmp -I/home/sharma/apps/forServer/boost_1_53_0_mt/boost_1_53_0/  -I/home/sharma/apps/forServer/eigen
LFLAGS = -L/home/sharma/apps/forServer/boost_1_53_0_mt/boost_1_53_0/stage/lib -lboost_serialization -lboost_mpi

SRC_cisd = CISD.cpp Hmult.cpp integral.cpp 
SRC_cipsi = CIPSI.cpp CIPSIbasics.cpp Hmult.cpp integral.cpp Davidson.cpp readInput.cpp
SRC_cipsi2 = CIPSI.cpp CIPSIbasics2.cpp Hmult.cpp integral.cpp Davidson.cpp readInput.cpp
SRC_Excitations = Excitations.cpp CIPSIbasics.cpp Hmult.cpp integral.cpp Davidson.cpp readInput.cpp

OBJ_cisd+=$(SRC_cisd:.cpp=.o)
OBJ_cipsi+=$(SRC_cipsi:.cpp=.o)
OBJ_Excitations+=$(SRC_Excitations:.cpp=.o)
OBJ_cipsi2+=$(SRC_cipsi2:.cpp=.o)

.cpp.o :
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@


all: CIPSI stats

stats: stats.o
	$(CXX) -O3 stats.cpp -o stats
CIPSI	: $(OBJ_cipsi)
	$(CXX)   $(FLAGS) $(OPT) -o  CIPSI $(OBJ_cipsi) $(LFLAGS)
Excitations	: $(OBJ_Excitations)
	$(CXX)   $(FLAGS) $(OPT) -o  Excitations $(OBJ_Excitations) $(LFLAGS)
CIPSI2	: $(OBJ_cipsi2)
	$(CXX)   $(FLAGS) $(OPT) -o  CIPSI2 $(OBJ_cipsi2) $(LFLAGS)
CISD	: $(OBJ_cisd)
	$(CXX)   $(FLAGS) $(OPT) -o  CISD $(OBJ_cisd) $(LFLAGS)

clean :
	rm *.o CISD CIPSI
