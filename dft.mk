CXX := mpic++
EIGEN := /home/junhao/eigen-3.3.4
BOOST := /home/junhao/boost-1.65.0
FLAGS := -std=c++17 -O3 -fopenmp -I $(EIGEN)/include -I $(BOOST)/include -DDFT
DFLAGS := $(FLAGS) -DComplex
LFLAGS := -L $(BOOST)/lib -lboost_serialization -lboost_mpi -lrt

