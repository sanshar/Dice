#include "global.h"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "Determinants.h"
#include "SHCImakeHamiltonian.h"
#include "SHCIrdm.h"
#include "input.h"
#include "integral.h"
#include "Hmult.h"
#include "SHCIbasics.h"
#include "Davidson.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <set>
#include <list>
#include <tuple>
#include "boost/format.hpp"
#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "communicate.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include "SOChelper.h"

using namespace Eigen;
using namespace boost;
int HalfDet::norbs = 1; //spin orbitals
int Determinant::norbs = 1; //spin orbitals
int Determinant::EffDetLen = 1;
Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> Determinant::LexicalOrder ;
//get the current time
double getTime() {
  struct timeval start;
  gettimeofday(&start, NULL);
  return start.tv_sec + 1.e-6*start.tv_usec;
}
double startofCalc = getTime();

boost::interprocess::shared_memory_object int2Segment;
boost::interprocess::mapped_region regionInt2;
boost::interprocess::shared_memory_object int2SHMSegment;
boost::interprocess::mapped_region regionInt2SHM;



void readInput(string input, vector<std::vector<int> >& occupied, schedule& schd);


int main(int argc, char* argv[]) {
#ifndef SERIAL
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif
  string inputFile = "input.dat";
  if (argc > 1)
    inputFile = string(argv[1]);
  //Read the input file
  std::vector<std::vector<int> > HFoccupied;
  schedule schd;
  if (mpigetrank() == 0) readInput(inputFile, HFoccupied, schd);
#ifndef SERIAL
  mpi::broadcast(world, HFoccupied, 0);
  mpi::broadcast(world, schd, 0);
#endif


  //set the random seed
  startofCalc=getTime();
  srand(schd.randomSeed+world.rank());
  //srand(schd.randomSeed+world.rank());
  if (mpigetrank()==0) cout<<"#using seed: "<<schd.randomSeed<<endl;



  //set up shared memory files to store the integrals
  string shciint2 = "SHCIint2" + to_string(static_cast<long long>(time(NULL) % 1000000));
  string shciint2shm = "SHCIint2shm" + to_string(static_cast<long long>(time(NULL) % 1000000));
  int2Segment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2.c_str(), boost::interprocess::read_write);
  int2SHMSegment = boost::interprocess::shared_memory_object(boost::interprocess::open_or_create, shciint2shm.c_str(), boost::interprocess::read_write);




  std::cout.precision(15);

  //read the hamiltonian (integrals, orbital irreps, num-electron etc.)
  twoInt I2; oneInt I1; int nelec; int norbs; double coreE=0.0, eps;
  std::vector<int> irrep;
  readIntegrals(schd.integralFile, I2, I1, nelec, norbs, coreE, irrep);
  
  if (HFoccupied[0].size() != nelec) {
    cout << "The number of electrons given in the FCIDUMP should be equal to the nocc given in the shci input file."<<endl;
    exit(0);
  }


  //setup the lexical table for the determinants
  norbs *=2;
  Determinant::norbs = norbs; //spin orbitals
  HalfDet::norbs = norbs; //spin orbitals
  Determinant::EffDetLen = norbs/64+1;
  Determinant::initLexicalOrder(nelec);
  if (Determinant::EffDetLen >DetLen) {
    cout << "change DetLen in global.h to "<<Determinant::EffDetLen<<" and recompile "<<endl;
    exit(0);
  }


  //initialize the heatbath integral
  std::vector<int> allorbs;
  for (int i=0; i<norbs/2; i++)
    allorbs.push_back(i);
  twoIntHeatBath I2HB(1.e-10);
  twoIntHeatBathSHM I2HBSHM(1.e-10);
  if (mpigetrank() == 0) I2HB.constructClass(allorbs, I2, norbs/2);
  I2HBSHM.constructClass(norbs/2, I2HB);

  int num_thrds;

  //IF SOC is true then read the SOC integrals
#ifndef Complex
  if (schd.doSOC) {
    cout << "doSOC option only works with the complex coefficients. Uncomment the -Dcomplex in the make file and recompile."<<endl;
    exit(0);
  }
#else
  if (schd.doSOC) {
    readSOCIntegrals(I1, norbs, "SOC");
    mpi::broadcast(world, I1, 0);
  }
#endif


  //have the dets, ci coefficient and diagnoal on all processors
  vector<MatrixXx> ci(schd.nroots, MatrixXx::Zero(HFoccupied.size(),1)); 
  vector<MatrixXx> vdVector(schd.nroots); //these vectors are used to calculate the response equations 
  double Psi1Norm = 0.0;

  //make HF determinant
  vector<Determinant> Dets(HFoccupied.size());
  for (int d=0;d<HFoccupied.size(); d++) {

    for (int i=0; i<HFoccupied[d].size(); i++) {
      Dets[d].setocc(HFoccupied[d][i], true);
    }
  }

  if (mpigetrank() == 0) {
    for (int j=0; j<ci[0].rows(); j++) 
      ci[0](j,0) = 1.0;
    ci[0] = ci[0]/ci[0].norm();
  }

  mpi::broadcast(world, ci, 0);



  vector<double> E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM, irrep, I1, coreE, nelec, schd.DoRDM);


  std::string efile;
  efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
  FILE* f = fopen(efile.c_str(), "wb");      
  for(int j=0;j<E0.size();++j) {
    fwrite( &E0[j], 1, sizeof(double), f);
  }
  fclose(f);



  //print the 5 most important determinants and their weights
  for (int root=0; root<schd.nroots; root++) {
    pout << "### IMPORTANT DETERMINANTS FOR STATE: "<<root<<endl;
    MatrixXx prevci = 1.*ci[root];
    for (int i=0; i<6; i++) {
      compAbs comp;
      int m = distance(&prevci(0,0), max_element(&prevci(0,0), &prevci(0,0)+prevci.rows(), comp));
      pout <<"#"<< i<<"  "<<prevci(m,0)<<"  "<<abs(prevci(m,0))<<"  "<<Dets[m]<<endl;
      prevci(m,0) = 0.0;
    }
  }
  pout << "### PERFORMING PERTURBATIVE CALCULATION"<<endl;


  if (schd.quasiQ) {    
    double bkpepsilon2 = schd.epsilon2;
    schd.epsilon2 = schd.quasiQEpsilon;
    for (int root=0; root<schd.nroots;root++) {
      E0[root] += SHCIbasics::DoPerturbativeDeterministic(Dets, ci[root], E0[root], I1, I2, I2HBSHM, irrep, schd, coreE, nelec, root, vdVector, Psi1Norm, true);
      ci[root] = ci[root]/ci[root].norm();
    }
    schd.epsilon2 = bkpepsilon2;
  }

  world.barrier();
  boost::interprocess::shared_memory_object::remove(shciint2.c_str());

  vector<MatrixXx> spinRDM(3, MatrixXx::Zero(norbs, norbs));
#ifdef Complex
  if (schd.doSOC) {
    for (int j=0; j<E0.size(); j++)
      cout << str(boost::format("State: %3d,  E: %17.9f, dE: %10.2f\n")%j %(E0[j]) %( (E0[j]-E0[0])*219470));

    //dont do this here, if perturbation theory is switched on
    if (schd.doGtensor)  {
      SOChelper::calculateSpinRDM(spinRDM, ci[0], ci[1], Dets, norbs, nelec);
      cout << "VARIATIONAL G-TENSOR"<<endl;
      SOChelper::doGTensor(ci, Dets, E0, norbs, nelec, spinRDM);
    }
  }
#endif


  if (schd.doSOC && !schd.stochastic) { //deterministic SOC calculation
    pout << "About to perform Perturbation theory"<<endl;
    if (schd.doGtensor) {
      pout << "Gtensor calculation not supported with deterministic PT for more than 2 roots."<<endl;
      pout << "Just performing the ZFS calculations."<<endl;
    }
    MatrixXx Heff = MatrixXx::Zero(E0.size(), E0.size());

    for (int root1 =0 ;root1<schd.nroots; root1++) {
      for (int root2=root1+1 ;root2<schd.nroots; root2++) {
	Heff(root1, root1) = 0.0; Heff(root2, root2) = 0.0; Heff(root1, root2) = 0.0;
	SHCIbasics::DoPerturbativeDeterministicOffdiagonal(Dets, ci[root1], E0[root1], ci[root2],
							   E0[root2], I1, 
							   I2, I2HBSHM, irrep, schd, 
							   coreE, nelec, root1, Heff(root1,root1),
							   Heff(root2, root2), Heff(root1, root2),
							   spinRDM);
	Heff(root2, root1) = conj(Heff(root1, root2));
      }
    }
    for (int root1 =0 ;root1<schd.nroots; root1++)
      Heff(root1, root1) += E0[root1];
    
    SelfAdjointEigenSolver<MatrixXx> eigensolver(Heff);
    for (int j=0; j<eigensolver.eigenvalues().rows(); j++) {
      E0[j] = eigensolver.eigenvalues()(j,0);
      pout << str(boost::format("State: %3d,  E: %17.9f, dE: %10.2f\n")%j %(eigensolver.eigenvalues()(j,0)) %( (eigensolver.eigenvalues()(j,0)-eigensolver.eigenvalues()(0,0))*219470));
    }

    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
    FILE* f = fopen(efile.c_str(), "wb");      
    for(int j=0;j<E0.size();++j) {
      fwrite( &E0[j], 1, sizeof(double), f);
    }
    fclose(f);

#ifdef Complex
    SOChelper::doGTensor(ci, Dets, E0, norbs, nelec, spinRDM);
    return 0;
#endif
  }
  
  else if (!schd.stochastic && schd.nblocks == 1) {
    double ePT = 0.0;
    std::string efile;
    efile = str(boost::format("%s%s") % schd.prefix[0].c_str() % "/shci.e" );
    FILE* f = fopen(efile.c_str(), "wb");      
    for (int root=0; root<schd.nroots;root++) {
      ePT = SHCIbasics::DoPerturbativeDeterministic(Dets, ci[root], E0[root], I1, I2,
						    I2HBSHM, irrep, schd, coreE, nelec,
						    root, vdVector, Psi1Norm);
      ePT += E0[root];
      fwrite( &ePT, 1, sizeof(double), f);
    }
    fclose(f);

  }
  else if (schd.SampleN != -1 && schd.singleList ){
    for (int root=0; root<schd.nroots && schd.nPTiter != 0;root++) 
      //SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2OMPTogether(Dets, ci[root], E0[root], I1, I2, I2HBSHM, irrep, schd, coreE, nelec, root);
      SHCIbasics::DoPerturbativeStochastic2SingleListDoubleEpsilon2AllTogether(Dets, ci[root], E0[root], I1, I2, I2HBSHM, irrep, schd, coreE, nelec, root);
      //SHCIbasics::DoPerturbativeStochastic2SingleList(Dets, ci[root], E0[root], I1, I2, I2HBSHM, irrep, schd, coreE, nelec, root);
  }
  else { 
    world.barrier();
    boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());
    cout << "Error here"<<endl;
    exit(0);
  }


  //THIS IS USED FOR RDM CALCULATION FOR DETERMINISTIC PT
  if ((schd.doResponse || schd.DoRDM) && (!schd.stochastic && schd.nblocks==1)) {
    std::vector<MatrixXx> lambda(schd.nroots, MatrixXx::Zero(Dets.size(),1));
    std::vector<std::vector<int> > connections;
    std::vector<std::vector<CItype> > Helements;
    std::vector<std::vector<size_t> > orbDifference;
    {
      char file [5000];
      sprintf (file, "%s/%d-hamiltonian.bkp" , schd.prefix[0].c_str(), world.rank() );
      std::ifstream ifs(file, std::ios::binary);
      boost::archive::binary_iarchive load(ifs);
      load >> connections >> Helements >> orbDifference;
    }


    Hmult2 H(connections, Helements);
    LinearSolver(H, E0[0], lambda[0], vdVector[0], ci, 1.e-5, false);
    mpi::broadcast(world, lambda[0], 0);

    MatrixXx s2RDM, twoRDM;
    s2RDM.setZero(norbs*norbs/4, norbs*norbs/4);
    if (schd.DoSpinRDM) twoRDM.setZero(norbs*(norbs+1)/2, norbs*(norbs+1)/2);
    SHCIrdm::EvaluateRDM(connections, Dets, lambda[0], ci[0], orbDifference, nelec, schd, 0, twoRDM, s2RDM);

    if (mpigetrank() == 0) {
      MatrixXx s2RDMdisk, twoRDMdisk;
      SHCIrdm::loadRDM(schd, s2RDMdisk, twoRDMdisk, 0);
      s2RDMdisk = s2RDMdisk + s2RDM.adjoint() + s2RDM;
      SHCIrdm::saveRDM(schd, s2RDMdisk, twoRDMdisk, 0);
    }
    //cout <<" response "; 
    //SHCIrdm::ComputeEnergyFromSpatialRDM(norbs, nelec, I1, I2, coreE, s2RDM);
  }

  world.barrier();
  boost::interprocess::shared_memory_object::remove(shciint2shm.c_str());

  return 0;
}
