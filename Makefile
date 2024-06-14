USE_INTEL = no
HAS_AVX2 = yes

BOOST=${BOOST_ROOT}
EIGEN=./eigen/
HDF5=${CURC_HDF5_ROOT}
MKL=${MKLROOT}

INCLUDE_MKL = -I$(MKL)/include
LIB_MKL = -lblas -llapack #-L$(MKL)/lib/intel64/ -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core

INCLUDE_BOOST = -I$(BOOST)/include #-I$(BOOST)
LIB_BOOST = -L$(BOOST)/lib -L$(BOOST)/stage/lib

INCLUDE_HDF5 = -I$(HDF5)/include
LIB_HDF5 = -L$(HDF5)/lib -lhdf5

COMPILE_NUMERIC = no

FLAGS_BASE = -std=c++14 -O3 -g -w -fPIC -I. -I$(EIGEN) $(INCLUDE_BOOST) $(INCLUDE_HDF5)
#FLAGS_BASE = -std=c++14 -g -w -fPIC -I. -I$(EIGEN) $(INCLUDE_BOOST) $(INCLUDE_HDF5)
LFLAGS_BASE = $(LIB_BOOST)
ifeq ($(HAS_AVX2), yes)
	FLAGS_BASE += -march=core-avx2
endif

ifeq ($(USE_INTEL), yes)
	LANG = en_US.utf8
	LC_ALL = en_US.utf8
	FLAGS_BASE += -qopenmp
	CXX = mpiicpc
else
	FLAGS_BASE += -fopenmp -fpermissive
	LFLAGS_BASE += -lrt
	CXX = mpicxx
endif

FLAGS_SHCI = $(FLAGS_BASE) -I./SHCI
FLAGS_ZDICE = $(FLAGS_BASE) -I./SHCI -DComplex
FLAGS_ZSHCI = $(FLAGS_BASE) -I./ZSHCI -DComplex
FLAGS_QMC = $(FLAGS_BASE) -I./VMC -I./utils -I./Wavefunctions  -I./FCIQMC 
FLAGS_ICPT = $(FLAGS_QMC) $(INCLUDE_MKL) -I./ICPT -I./ICPT/StackArray

LFLAGS_SHCI = $(LFLAGS_BASE) -lboost_mpi -lboost_serialization
LFLAGS_ZDICE = $(LFLAGS_SHCI)
LFLAGS_ZSHCI = $(LFLAGS_SHCI)
LFLAGS_QMC = $(LFLAGS_SHCI) $(LIB_HDF5) -lboost_program_options -lboost_system -lboost_filesystem
LFLAGS_ICPT = $(LFLAGS_QMC) $(LIB_MKL)

GIT_HASH = `git rev-parse HEAD`
COMPILE_TIME = `date`
GIT_BRANCH = `git branch | grep "^\*" | sed s/^..//`
VERSION_FLAGS = -DGIT_HASH="\"$(GIT_HASH)\"" -DCOMPILE_TIME="\"$(COMPILE_TIME)\"" \
	-DGIT_BRANCH="\"$(GIT_BRANCH)\""

# Host specific configurations.
HOSTNAME := $(shell hostname)
ifneq ($(filter dft node%, $(HOSTNAME)),)
include dft.mk
endif

OBJ_VMC = obj/VMC.o \
    obj/staticVariables.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Slater.o \
	obj/MultiSlater.o \
	obj/AGP.o \
	obj/Pfaffian.o \
	obj/Jastrow.o \
	obj/SJastrow.o \
	obj/Gutzwiller.o \
	obj/CPS.o \
	obj/RBM.o \
	obj/JRBM.o \
	obj/Correlator.o \
	obj/SelectedCI.o \
	obj/SCPT.o \
	obj/SimpleWalker.o \
	obj/ShermanMorrisonWoodbury.o\
	obj/excitationOperators.o\
	obj/statistics.o \
	obj/sr.o \
	obj/evaluateE.o 

OBJ_ICPT= obj/PerturberDependentCode.o \
	obj/BlockContract.o \
	obj/CxAlgebra.o \
	obj/CxIndentStream.o \
	obj/CxNumpyArray.o \
	obj/icpt.o \
	obj/CxMemoryStack.o \
	obj/CxStorageDevice.o \
	obj/TensorTranspose.o \
	obj/ICPT.o

OBJ_GFMC = obj/GFMC.o \
    obj/staticVariables.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Slater.o \
	obj/AGP.o \
	obj/Pfaffian.o \
	obj/Jastrow.o \
	obj/Gutzwiller.o \
	obj/CPS.o \
	obj/evaluateE.o \
	obj/excitationOperators.o\
	obj/ShermanMorrisonWoodbury.o\
	obj/statistics.o \
	obj/sr.o \
	obj/Correlator.o

OBJ_FCIQMC = obj/FCIQMC.o \
    obj/staticVariables.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Correlator.o \
	obj/runFCIQMC.o \
	obj/spawnFCIQMC.o \
	obj/walkersFCIQMC.o \
	obj/excitGen.o \
	obj/utilsFCIQMC.o \
	obj/Slater.o \
	obj/AGP.o \
	obj/Jastrow.o \
	obj/SelectedCI.o \
	obj/SimpleWalker.o \
	obj/ShermanMorrisonWoodbury.o \
	obj/excitationOperators.o \
	obj/statistics.o \
	obj/sr.o \
	obj/evaluateE.o

OBJ_DQMC = obj/DQMC.o \
    obj/staticVariables.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Correlator.o \
	obj/DQMCMatrixElements.o \
	obj/DQMCStatistics.o \
	obj/DQMCWalker.o \
	obj/Hamiltonian.o \
	obj/RHF.o \
	obj/UHF.o \
	obj/GHF.o \
	obj/GZHF.o \
	obj/KSGHF.o \
	obj/Multislater.o \
	obj/GHFMultislater.o \
	obj/GZHFMultislater.o \
	obj/CCSD.o \
	obj/UCCSD.o \
	obj/sJastrow.o \
	obj/MixedEstimator.o \
	obj/ProjectedMF.o

OBJ_Dice = \
	obj/SHCI/SHCIbasics.o \
	obj/SHCI/Determinants.o \
	obj/SHCI/integral.o \
	obj/SHCI/input.o \
	obj/SHCI/Davidson.o \
	obj/SHCI/SHCIgetdeterminants.o \
	obj/SHCI/SHCIsampledeterminants.o \
	obj/SHCI/SHCIrdm.o \
	obj/SHCI/SHCISortMpiUtils.o \
	obj/SHCI/SHCImakeHamiltonian.o \
	obj/SHCI/SHCIshm.o \
	obj/SHCI/LCC.o \
	obj/SHCI/symmetry.o \
	obj/SHCI/OccRestrictions.o \
	obj/SHCI/cdfci.o

OBJ_ZDice2 = \
	obj_z/SHCI/SHCI.o \
	obj_z/SHCI/SHCIbasics.o \
	obj_z/SHCI/Determinants.o \
	obj_z/SHCI/integral.o \
	obj_z/SHCI/input.o \
	obj_z/SHCI/Davidson.o \
	obj_z/SHCI/SOChelper.o \
	obj_z/SHCI/new_anglib.o \
	obj_z/SHCI/SHCIgetdeterminants.o \
	obj_z/SHCI/SHCIsampledeterminants.o \
	obj_z/SHCI/SHCIrdm.o \
	obj_z/SHCI/SHCISortMpiUtils.o \
	obj_z/SHCI/SHCImakeHamiltonian.o \
	obj_z/SHCI/SHCIshm.o \
	obj_z/SHCI/LCC.o \
	obj_z/SHCI/symmetry.o \
	obj/SHCI/OccRestrictions.o

OBJ_ZSHCI = \
	obj_z/ZSHCI/SHCI.o \
	obj_z/ZSHCI/SHCIbasics.o \
	obj_z/ZSHCI/Determinants.o \
	obj_z/ZSHCI/integral.o \
	obj_z/ZSHCI/input.o \
	obj_z/ZSHCI/Davidson.o \
	obj_z/ZSHCI/SOChelper.o \
	obj_z/ZSHCI/new_anglib.o \
	obj_z/ZSHCI/SHCIgetdeterminants.o \
	obj_z/ZSHCI/SHCIsampledeterminants.o \
	obj_z/ZSHCI/SHCIrdm.o \
	obj_z/ZSHCI/SHCISortMpiUtils.o \
	obj_z/ZSHCI/SHCImake4cHamiltonian.o \
	obj_z/ZSHCI/SHCIshm.o \
	obj_z/ZSHCI/LCC.o \
	obj_z/ZSHCI/symmetry.o \
	obj_z/ZSHCI/cdfci.o

obj/SHCI/%.o: SHCI/%.cpp
	$(CXX) $(FLAGS_SHCI) $(OPT) $(VERSION_FLAGS) -c $< -o $@
obj_z/SHCI/%.o: SHCI/%.cpp
	$(CXX) $(FLAGS_ZDICE) $(OPT) $(VERSION_FLAGS) -c $< -o $@
obj_z/ZSHCI/%.o: ZSHCI/%.cpp
	$(CXX) $(FLAGS_ZSHCI) $(OPT) $(VERSION_FLAGS) -c $< -o $@
obj/%.o: Wavefunctions/%.cpp  
	$(CXX) $(FLAGS_QMC) $(OPT) $(VERSION_FLAGS) -c $< -o $@
obj/%.o: utils/%.cpp  
	$(CXX) $(FLAGS_QMC) $(OPT) $(VERSION_FLAGS) -c $< -o $@
obj/%.o: VMC/%.cpp  
	$(CXX) $(FLAGS_QMC) -I./VMC $(OPT) $(VERSION_FLAGS) -c $< -o $@
obj/%.o: DQMC/%.cpp  
	$(CXX) $(FLAGS_QMC) -I./DQMC $(OPT) $(VERSION_FLAGS) -c $< -o $@
obj/%.o: FCIQMC/%.cpp  
	$(CXX) $(FLAGS_QMC) -I./FCIQMC $(OPT) $(VERSION_FLAGS) -c $< -o $@
obj/%.o: ICPT/%.cpp  
	$(CXX) $(FLAGS_ICPT) -I./ICPT/TensorExpressions/ $(OPT) $(VERSION_FLAGS) -c $< -o $@
obj/%.o: ICPT/StackArray/%.cpp  
	$(CXX) $(FLAGS_ICPT) $(OPT) $(VERSION_FLAGS) -c $< -o $@
obj/%.o: executables/%.cpp  
	$(CXX) $(FLAGS_QMC) -I./SHCI -I./VMC -I./DQMC -I./FCIQMC -I./GFMC -I./ICPT/StackArray -I./ICPT $(OPT) $(VERSION_FLAGS) -c $< -o $@

# not sure about the status of periodic
#ALL= bin/VMC bin/GFMC bin/ICPT bin/FCIQMC bin/DQMC
#ifeq ($(COMPILE_NUMERIC), yes)
#  ALL+= bin/periodic
#endif 

# all: VMC GFMC FCIQMC DQMC ICPT Dice ZDice2 ZSHCI
all: VMC GFMC DQMC ICPT Dice ZDice2 ZSHCI alllibs lib/libSHCIPostProcess.so

alllibs:
	gcc -O3 -fPIC -fopenmp -shared lib/icmpspt.c -o lib/libicmpspt.so
	gcc -O3 -fPIC -fopenmp -shared lib/shcitools.c -o lib/libSHCITools.so

lib/libSHCIPostProcess.so: $(OBJ_Dice) obj/SHCI/postprocess.o
	$(CXX) -shared $(FLAGS_SHCI) $(OPT) -o lib/libSHCIPostProcess.so $(OBJ_Dice) obj/SHCI/postprocess.o $(LFLAGS_SHCI) -I./SHCI

periodic: 
	cd ./NumericPotential/PeriodicIntegrals/ && $(MAKE) -f Makefile && cp a.out ../../bin/periodic

bin/libPeriodic.so: bin/libPeriodic.so
	cd ./NumericPotential/ && $(MAKE) -f Makefile

GFMC: $(OBJ_GFMC) 
	$(CXX) $(FLAGS_QMC) $(OPT) -o bin/GFMC $(OBJ_GFMC)  $(LFLAGS_QMC) $(VERSION_FLAGS)

ICPT: $(OBJ_ICPT) 
	$(CXX) $(FLAGS_ICPT) $(OPT) -o bin/ICPT $(OBJ_ICPT)  $(LFLAGS_ICPT) $(VERSION_FLAGS)

VMC: $(OBJ_VMC)
	$(CXX) $(FLAGS_QMC) $(OPT) -o  bin/VMC $(OBJ_VMC) $(LFLAGS_QMC) $(VERSION_FLAGS)

FCIQMC: $(OBJ_FCIQMC) 
	$(CXX) $(FLAGS_QMC) $(OPT) -o bin/FCIQMC $(OBJ_FCIQMC) $(LFLAGS_QMC) $(VERSION_FLAGS)

DQMC: $(OBJ_DQMC) 
	$(CXX) $(FLAGS_QMC) $(OPT) -o  bin/DQMC $(OBJ_DQMC) $(LFLAGS_QMC) $(VERSION_FLAGS)

Dice	: $(OBJ_Dice) obj/SHCI/SHCI.o
	$(CXX) $(FLAGS_SHCI) $(OPT) -o bin/Dice $(OBJ_Dice) obj/SHCI/SHCI.o $(LFLAGS_SHCI) -I./SHCI

ZDice2	: $(OBJ_ZDice2)
	$(CXX) $(FLAGS_ZDICE) $(OPT) -o bin/ZDice2 $(OBJ_ZDice2) $(LFLAGS_ZDICE) -I./SHCI -DComplex

ZSHCI	: $(OBJ_ZSHCI)
	$(CXX) $(FLAGS_ZSHCI) $(OPT) -o bin/ZSHCI $(OBJ_ZSHCI) $(LFLAGS_ZSHCI)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm -f bin/* >/dev/null 2>&1


