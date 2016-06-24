#include "Determinants.h"
#include "CIPSIbasics.h"
#include "integral.h"
#include <vector>
#include "math.h"
#include "Hmult.h"
#include <tuple>
#include <map>
#include "Davidson.h"
#include "boost/format.hpp"
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using namespace std;
using namespace Eigen;
using namespace boost;

//for each element in ci stochastic round to eps and put all the nonzero elements in newWts and their corresponding
//indices in Sample1
int sample_round(MatrixXd& ci, double eps, std::vector<int>& Sample1, std::vector<double>& newWts){
  for (int i=0; i<ci.rows(); i++) {
    if (abs(ci(i,0)) > eps) {
      Sample1.push_back(i);
      newWts.push_back(ci(i,0));
    }
    else if (((double) rand() / (RAND_MAX))*eps < abs(ci(i,0))) {
      Sample1.push_back(i);
      newWts.push_back( eps*ci(i,0)/abs(ci(i,0)));
    }
  }
}

int sample_N(MatrixXd& ci, double& cumulative, std::vector<int>& Sample1, std::vector<double>& newWts){
  double prob = 1.0;
  for (int s=0; s<Sample1.size(); s++) {

    double rand_no = ((double) rand() / (RAND_MAX))*cumulative;
    for (int i=0; i<ci.rows(); i++) {
      if (rand_no < abs(ci(i,0))) {
	Sample1[s] = i;
	newWts[s] = ci(i,0) < 0. ? -cumulative/Sample1.size() : cumulative/Sample1.size();
	break;
      }
      rand_no -= abs(ci(i,0));
    }
  }

}


void CIPSIbasics::DoDeterministic2(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
				   twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {
  int nblocks = schd.nblocks;
  std::vector<int> blockSizes(nblocks,0);
  for (int i=0; i<nblocks; i++) {
    if (i!=nblocks-1)
      blockSizes[i] = Dets.size()/nblocks;
    else
      blockSizes[i] = Dets.size() - (nblocks-1)*Dets.size()/(nblocks);
  }
  int norbs = Determinant::norbs;
  std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
  double AvgenergyEN = 0.0;

#pragma omp parallel for schedule(dynamic) 
  for (int inter1 = 0; inter1 < nblocks; inter1++) {

    vector<int> psiClosed(nelec,0); 
    vector<int> psiOpen(norbs-nelec,0);

    std::vector<double> wts1(blockSizes[inter1]);
    std::vector<int> Sample1(blockSizes[inter1]);
    for (int i=0; i<wts1.size(); i++) {
      wts1[i] = ci(i+(inter1)*Dets.size()/nblocks ,0);
      Sample1[i] = i+(inter1)*Dets.size()/nblocks ;
    }

    map<Determinant, pair<double,double> > Psi1ab; 
    for (int i=0; i<Sample1.size(); i++) {
      int I = Sample1[i];
      CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, 0);
    }

    double energyEN = 0.0;
    for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      energyEN += it->second.first*it->second.first/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
    }
    

    for (int i=Sample1[Sample1.size()-1]+1; i<Dets.size(); i++) {
      CIPSIbasics::getDeterminants(Dets[i], abs(schd.epsilon2/ci(i,0)), ci(i,0), I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, 1);
      if (i%1000 == 0 && omp_get_thread_num() == 0) cout <<i <<" out of "<<Dets.size()-Sample1.size()<<endl; 
    }

    for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
      it->first.getOpenClosed(psiOpen, psiClosed);
      energyEN += 2.*it->second.first*it->second.second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
    }

#pragma omp critical 
    {
      AvgenergyEN += energyEN;
      
      std::cout << format("%4i  %14.8f   %10.2f  %10i %4i") 
	%(inter1) % (E0-AvgenergyEN) % (getTime()-startofCalc) % inter1 % (omp_get_thread_num());
      cout << endl;
    }
  }
  
  {
    std::cout <<"FINAL ANSWER "<<endl;
    std::cout << format("%4i  %14.8f   %10.2f  %10i %4i") 
      %(nblocks) % (E0-AvgenergyEN) % (getTime()-startofCalc) % nblocks % (0);
    cout << endl;
  }

}



void CIPSIbasics::DoPerturbativeStochastic2(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					   twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {
    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 10000;
    //double eps = 0.001;
    int Nsample = schd.SampleN;
    double AvgenergyEN = 0.0;
    double AverageDen = 0.0;
    int currentIter = 0;
    int sampleSize = 0;
    int num_thrds = omp_get_max_threads();

    double cumulative = 0.0;
    for (int i=0; i<ci.rows(); i++)
      cumulative += abs(ci(i,0));

#pragma omp parallel for schedule(dynamic) 
    for (int iter=0; iter<niter; iter++) {
      //cout << norbs<<"  "<<nelec<<endl;
      char psiArray[norbs]; 
      vector<int> psiClosed(nelec,0); 
      vector<int> psiOpen(norbs-nelec,0);
      //char psiArray[norbs];
      std::vector<double> wts1(Nsample), wts2(Nsample); std::vector<int> Sample1(Nsample), Sample2(Nsample);
      //wts1.reserve(Nsample); wts2.reserve(Nsample); Sample1.reserve(Nsample); Sample2.reserve(Nsample);
      

      sample_N(ci, cumulative, Sample1, wts1);
      sample_N(ci, cumulative, Sample2, wts2);

      double norm = 0.0;
      for (int i=0; i<Sample1.size(); i++) {
	double normi = 0.0;
	for (int j=0; j<Sample2.size(); j++)
	  if (Sample2[j] == Sample1[i]) normi += wts1[i]*wts2[j];
	norm += normi;
      }
      
      map<Determinant, pair<double,double> > Psi1ab; 
      for (int i=0; i<Sample1.size(); i++) {
	int I = Sample1[i];
	CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, 0);
      }


      for (int i=0; i<Sample2.size(); i++) {
	int I = Sample2[i];
	CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts2[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, 1);
      }


      double energyEN = 0.0;
      for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
	it->first.getOpenClosed(psiOpen, psiClosed);
	energyEN += it->second.first*it->second.second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
      }
      sampleSize = Sample1.size();
      AverageDen += norm;
#pragma omp critical 
      {
	AvgenergyEN += energyEN; currentIter++;
	std::cout << format("%4i  %14.8f  %14.8f %14.8f   %10.2f  %10i %4i") 
	  %(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	cout << endl;
      }
    }

}


void CIPSIbasics::DoPerturbativeStochastic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					   twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {
    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    int niter = 10000;
    //double eps = 0.001;
    double AvgenergyEN = 0.0;
    double AverageDen = 0.0;
    int currentIter = 0;
    int sampleSize = 0;
    int num_thrds = omp_get_max_threads();

#pragma omp parallel for schedule(dynamic) 
    for (int iter=0; iter<niter; iter++) {
      //cout << norbs<<"  "<<nelec<<endl;
      char psiArray[norbs]; 
      vector<int> psiClosed(nelec,0); 
      vector<int> psiOpen(norbs-nelec,0);
      //char psiArray[norbs];
      std::vector<double> wts1, wts2; std::vector<int> Sample1, Sample2;
      wts1.reserve(1000); wts2.reserve(1000); Sample1.reserve(1000); Sample2.reserve(1000);
      
      Sample1.resize(0); wts1.resize(0); Sample2.resize(0); wts2.resize(0);
      sample_round(ci, schd.eps, Sample1, wts1);
      sample_round(ci, schd.eps, Sample2, wts2);

      double norm = 0.0;
      for (int i=0; i<Sample1.size(); i++) {
	for (int j=0; j<Sample2.size(); j++)
	  if (Sample2[j] == Sample1[i]) {
	    norm += wts1[i]*wts2[j];
	    break;
	  }
      }
      
      map<Determinant, pair<double,double> > Psi1ab; 
      for (int i=0; i<Sample1.size(); i++) {
	int I = Sample1[i];
	CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts1[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, 0);
      }


      for (int i=0; i<Sample2.size(); i++) {
	int I = Sample2[i];
	CIPSIbasics::getDeterminants(Dets[I], abs(schd.epsilon2/ci(I,0)), wts2[i], I1, I2, I2HB, irrep, coreE, E0, Psi1ab, SortedDets, 1);
      }


      double energyEN = 0.0;
      for (map<Determinant, pair<double, double> >::iterator it = Psi1ab.begin(); it != Psi1ab.end(); it++) {
	it->first.getOpenClosed(psiOpen, psiClosed);
	energyEN += it->second.first*it->second.second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
      }
      sampleSize = Sample1.size();

#pragma omp critical 
      {
	AverageDen += norm;
	AvgenergyEN += energyEN; currentIter++;
	std::cout << format("%4i  %14.8f  %14.8f  %14.8f   %10.2f  %10i %4i") 
	  %(currentIter) % (E0-energyEN) % (norm) % (E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	cout << endl;
	//%(currentIter) % (E0-AvgenergyEN/currentIter) % (norm) % (E0-AvgenergyEN/AverageDen) % (getTime()-startofCalc) % sampleSize % (omp_get_thread_num());
	//cout << endl;
      }
    }

}


void CIPSIbasics::DoPerturbativeDeterministic(vector<Determinant>& Dets, MatrixXd& ci, double& E0, oneInt& I1, twoInt& I2, 
					      twoIntHeatBath& I2HB, vector<int>& irrep, schedule& schd, double coreE, int nelec) {

    int norbs = Determinant::norbs;
    std::vector<Determinant> SortedDets = Dets; std::sort(SortedDets.begin(), SortedDets.end());
    char psiArray[norbs]; vector<int> psiClosed(nelec,0), psiOpen(norbs-nelec,0);
    //char psiArray[norbs]; int psiOpen[nelec], psiClosed[norbs-nelec];
    double energyEN = 0.0;
    int num_thrds = omp_get_max_threads();

    std::vector<std::map<Determinant, double>> Psi1(num_thrds);
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<Dets.size(); i++) {
      CIPSIbasics::getDeterminants(Dets[i], abs(schd.epsilon2/ci(i,0)), ci(i,0), I1, I2, I2HB, irrep, coreE, E0, Psi1[omp_get_thread_num()], SortedDets);
      if (i%1000 == 0 && omp_get_thread_num()==0) cout <<"# "<<i<<endl;
    }

    for (int thrd=1; thrd<num_thrds; thrd++) 
      for (map<Determinant, double>::iterator it=Psi1[thrd].begin(); it!=Psi1[thrd].end(); ++it)  {
	if(Psi1[0].find(it->first) == Psi1[0].end())
	  Psi1[0][it->first] = it->second;
	else
	  Psi1[0][it->first] += it->second;
      }
    Psi1.resize(1);


    //cout << "adding contributions from "<<Psi1[0].size()<<" perturber states"<<endl;

#pragma omp parallel
    {
      vector<int> psiOpen(norbs-nelec,0), psiClosed(nelec,0);
      double thrdEnergy = 0.0;
      size_t cnt = 0;
      for (map<Determinant, double>::iterator it = Psi1[0].begin(); it != Psi1[0].end(); it++, cnt++) {
	if (cnt%num_thrds == omp_get_thread_num()) {
	  it->first.getOpenClosed(psiOpen, psiClosed);
	  thrdEnergy += it->second*it->second/(Energy(psiClosed, nelec, I1, I2, coreE)-E0); 
	}
      }
#pragma omp critical
      {
	cout << omp_get_thread_num()<<"  "<<thrdEnergy<<endl;
	energyEN += thrdEnergy;
      }
    }

    cout <<energyEN<<"  "<< -energyEN+E0<<"  "<<getTime()-startofCalc<<endl;

}


//this takes in a ci vector for determinants placed in Dets
//it then does a CIPSI varitional calculation and the resulting
//ci and dets are returned here
//at input usually the Dets will just have a HF or some such determinant
//and ci will be just 1.0
double CIPSIbasics::DoVariational(MatrixXd& ci, vector<Determinant>& Dets, schedule& schd,
				  twoInt& I2, twoIntHeatBath& I2HB, oneInt& I1, double& coreE) {

  int num_thrds = omp_get_max_threads();

  //make a copy of the input Dets and sort them but remember
  //where in the original Dets list the current Det belongs 
  std::map<Determinant,int> SortedDets; SortedDets[Dets[0]]=0;

  size_t norbs = 2.*I2.Direct.rows();
  int Norbs = norbs;
  std::vector<char> detChar(norbs); Dets[0].getRepArray(&detChar[0]);
  double E0 = Energy(&detChar[0], Norbs, I1, I2, coreE);
  std::cout << "#HF = "<<E0<<std::endl;

  //this is essentially the hamiltonian, we have stored it in a sparse format
  std::vector<std::vector<int> > connections(1, std::vector<int>(1,0));
  std::vector<std::vector<double> > Helements(1, std::vector<double>(1,E0));

  //keep the diagonal energies of determinants so we only have to generated
  //this for the new determinants in each iteration and not all determinants
  MatrixXd diagOld(1,1); diagOld(0,0) = E0;
  int prevSize = 0;

  int iterstart = 0;

  if (schd.restart || schd.fullrestart) {
    bool converged;
    readVariationalResult(iterstart, ci, Dets, SortedDets, diagOld, connections, Helements, E0, converged, schd.prefix);
    std::cout << format("# %4i  %10.2e  %10.2e   %14.8f  %10.2f\n") 
      %(iterstart) % schd.epsilon1[iterstart] % Dets.size() % E0 % (getTime()-startofCalc);
    iterstart++;
    detChar.resize(Dets.size()*norbs);
    for (int i=0; i<Dets.size(); i++) 
      Dets[i].getRepArray(&detChar[i*norbs]);
    if (converged && !schd.fullrestart) {
      cout << "# restarting from a converged calculation, moving to perturbative part.!!"<<endl;
      return E0;
    }
  }

  //do the variational bit
  for (int iter=iterstart; iter<schd.epsilon1.size(); iter++) {
    double epsilon1 = schd.epsilon1[iter];
    std::vector<set<Determinant> > newDets(num_thrds);
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<SortedDets.size(); i++) {
      getDeterminants(Dets[i], abs(epsilon1/ci(i,0)), I1, I2, I2HB, coreE, E0, newDets[omp_get_thread_num()]);
      if (i%1000000 == 0 && i!=0) cout <<"#"<< i<<" out of "<<SortedDets.size()<<endl;
    }

    for (int thrd=1; thrd<num_thrds; thrd++) {
      for (set<Determinant>::iterator it=newDets[thrd].begin(); it!=newDets[thrd].end(); ++it) 
	if(newDets[0].find(*it) == newDets[0].end())
	  newDets[0].insert(*it);
      newDets[thrd].clear();
    }

    for (set<Determinant>::iterator it=newDets[0].begin(); it!=newDets[0].end(); ++it) 
      if (SortedDets.find(*it) == SortedDets.end())
	Dets.push_back(*it);
    newDets[0].clear();
    
    //now diagonalize the hamiltonian
    detChar.resize(norbs* Dets.size()); 
    MatrixXd X0(Dets.size(), 1); X0.setZero(Dets.size(),1); X0.block(0,0,ci.rows(),1) = 1.*ci; 
    MatrixXd diag(Dets.size(), 1); diag.block(0,0,ci.rows(),1)= 1.*diagOld;

#pragma omp parallel for schedule(dynamic)
    for (size_t k=SortedDets.size(); k<Dets.size(); k++) {
      Dets[k].getRepArray(&detChar[norbs*k]);
      diag(k,0) = Energy(&detChar[norbs*k], Norbs, I1, I2, coreE);
      if (k%1000000 == 0 && k!=0) cout <<"#"<< k<<"Hdiag out of "<<Dets.size()<<endl;     
    }
    //cout << "diag"<<endl;

    //update connetions
    connections.resize(Dets.size());
    Helements.resize(Dets.size());


    //if (Dets.size() > 10000000)
    if (false)
      updateConnections(Dets, SortedDets, norbs, I1, I2, coreE, &detChar[0], connections, Helements);
    //update connetions
    else {
#pragma omp parallel for schedule(dynamic)
      for (size_t i=0; i<Dets.size(); i++) {
	for (size_t j=max(SortedDets.size(),i); j<Dets.size(); j++) {
	  if (Dets[i].connected(Dets[j])) {
	    double hij = Hij(&detChar[norbs*i], &detChar[norbs*j], Norbs, I1, I2, coreE);
	    //double hij = Hij(Dets[i], Dets[j], norbs, I1, I2, coreE);
	    if (abs(hij) > 1.e-10) {
	      connections[i].push_back(j);
	      Helements[i].push_back(hij);
	    }
	  }
	}
	if (i%1000000 == 0 && i!=0) cout <<"#"<< i<<" Hij out of "<<Dets.size()<<endl;     
	//if (i%10000) cout << i<<"  "<<Dets.size()<<endl;
      }

      for (size_t i=SortedDets.size(); i<Dets.size(); i++)
	SortedDets[Dets[i]] = i;
    }

    double prevE0 = E0;
    //Hmult H(&detChar[0], norbs, I1, I2, coreE);
    Hmult2 H(connections, Helements);
    E0 = davidson(H, X0, diag, 5, schd.davidsonTol, false);
    std::cout << format("# %4i  %10.2e  %10.2e   %14.8f  %10.2f") 
      %(iter) % epsilon1 % Dets.size() % E0 % (getTime()-startofCalc);
    std::cout << endl;

    ci.resize(Dets.size(),1); ci = 1.0*X0;
    diagOld.resize(Dets.size(),1); diagOld = 1.0*diag;

    
    if (abs(E0-prevE0) < schd.dE)  {
      writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, Helements, E0, true, schd.prefix);
      break;
    }
    else 
      writeVariationalResult(iter, ci, Dets, SortedDets, diag, connections, Helements, E0, false, schd.prefix);



  }
  return E0;

}

void CIPSIbasics::writeVariationalResult(int iter, MatrixXd& ci, vector<Determinant>& Dets, map<Determinant,int>& SortedDets,
					 MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
					 double& E0, bool converged, string& prefix) {
  char file [5000];
  sprintf (file, "%s/variational.bkp" , prefix.c_str() );
  std::ofstream ofs(file, std::ios::binary);
  boost::archive::binary_oarchive save(ofs);

  save << iter <<Dets<<SortedDets<<connections<<Helements;
  int diagrows = diag.rows();
  save << diagrows;
  for (int i=0; i<diag.rows(); i++)
    save << diag(i,0);
  for (int i=0; i<ci.rows(); i++)
    save << ci(i,0);
  save << E0;
  save<<converged;
  ofs.close();
}


void CIPSIbasics::readVariationalResult(int& iter, MatrixXd& ci, vector<Determinant>& Dets, map<Determinant,int>& SortedDets,
					MatrixXd& diag, vector<vector<int> >& connections, vector<vector<double> >& Helements, 
					double& E0, bool& converged, string& prefix) {
  char file [5000];
  sprintf (file, "%s/variational.bkp" , prefix.c_str() );
  std::ifstream ifs(file, std::ios::binary);
  boost::archive::binary_iarchive load(ifs);

  load >> iter >> Dets >> SortedDets >> connections >> Helements;
  int diaglen;
  load >> diaglen;
  ci.resize(diaglen,1); diag.resize(diaglen,1);
  for (int i=0; i<diag.rows(); i++)
    load >> diag(i,0);
  for (int i=0; i<ci.rows(); i++)
    load >>  ci(i,0);
  load >> E0;
  load >> converged;
  ifs.close();
}



void CIPSIbasics::updateConnections(vector<Determinant>& Dets, map<Determinant, int>& SortedDets, int norbs, oneInt& int1, twoInt& int2, double coreE, char* detArray, vector<vector<int> >& connections, vector<vector<double> >& Helements) {
  size_t prevSize = SortedDets.size();
  size_t Norbs = norbs;
  for (size_t i=prevSize; i<Dets.size(); i++) {
    SortedDets[Dets[i]] = i;
    connections[i].push_back(i);
    Helements[i].push_back(Energy(&detArray[i*Norbs], norbs, int1, int2, coreE));
  }

#pragma omp parallel for schedule(dynamic)
  for (size_t x=prevSize; x<Dets.size(); x++) {
    Determinant d = Dets[x];
    int open[norbs], closed[norbs]; 
    int nclosed = d.getOpenClosed(open, closed);
    int nopen = norbs-nclosed;
    
    if (x%100000 == 0) cout <<"update connections "<<x<<" out of "<<Dets.size()-prevSize<<endl;
    //loop over all single excitation and find if they are present in the list
    //on or before the current determinant
    for (int ia=0; ia<nopen*nclosed; ia++){
      int i=ia/nopen, a=ia%nopen;
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      
      map<Determinant, int>::iterator it = SortedDets.find(di);
      if (it != SortedDets.end()) {
	int y = it->second;
	if (y <= x) { //avoid double counting
	  double integral = Hij_1Excite(closed[i],open[a],int1,int2, &detArray[x*Norbs], norbs);
	  if (abs(integral) > 1.e-8) {
	    connections[x].push_back(y);
	    Helements[x].push_back(integral);
	  }
	  //connections[y].push_back(x);
	  //Helements[y].push_back(integral);
	}
      }
    }


    for (int i=0; i<nclosed; i++)
      for (int j=0; j<i; j++) {
	for (int a=0; a<nopen; a++){
	  for (int b=0; b<a; b++){
	    Determinant di = d;
	    di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);

	    map<Determinant, int>::iterator it = SortedDets.find(di);
	    if (it != SortedDets.end()) {
	      int y = it->second;
	      if (y <= x) { //avoid double counting
		double integral = Hij_2Excite(closed[i], closed[j], open[a], open[b], int2, &detArray[x*Norbs], norbs);
		if (abs(integral) > 1.e-8) {
		  connections[x].push_back(y);
		  Helements[x].push_back(integral);
		  //cout << x<<"  "<<y<<"  "<<integral<<endl;
		}
		//connections[y].push_back(x);
		//Helements[y].push_back(integral);
	      }
	    }
	  }
	}
      }
  }

  
}

  
void CIPSIbasics::getDeterminants(Determinant& d, double epsilon, double ci, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, double >& dets, std::vector<Determinant>& Psi0, bool additions) {
  
  int norbs = d.norbs;
  int open[norbs], closed[norbs]; char detArray[norbs], diArray[norbs];
  int nclosed = d.getOpenClosed(open, closed);
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);
  
  std::map<Determinant, double>::iterator det_it;
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;
    double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);
    if (fabs(integral) > epsilon ) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {
	det_it = dets.find(di);

	if (additions) {
	  if (det_it == dets.end()) dets[di] = integral*ci;
	  else det_it->second +=integral*ci;
	}
	else 
	  if (det_it != dets.end()) det_it->second +=integral*ci;

      }
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    //THERE IS A BUG IN THE CODE WHEN USING HEATBATH INTEGRALS
    if (true && (ints != I2hb.sameSpin.end() || ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (!(d.getocc(a) || d.getocc(b))) {
	  Determinant di = d;
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    
	  if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {

	    double sgn = 1.0;
	    {
	      int A = (closed[i]), B = closed[j], I= a, J = b; 
	      sgn = parity(detArray,norbs,A)*parity(detArray,norbs,I)*parity(detArray,norbs,B)*parity(detArray,norbs,J);
	      if (B > J) sgn*=-1 ;
	      if (I > J) sgn*=-1 ;
	      if (I > B) sgn*=-1 ;
	      if (A > J) sgn*=-1 ;
	      if (A > B) sgn*=-1 ;
	      if (A > I) sgn*=-1 ;
	    }

	    det_it = dets.find(di);
	    if (additions) {
	      if (det_it == dets.end()) dets[di] = it->first*ci*sgn;
	      else det_it->second +=it->first*ci*sgn;
	    }
	    else 
	      if (det_it != dets.end()) det_it->second +=it->first*ci*sgn;

	  }
	}
      }
    }
    else {
      for (int a=0; a<nopen; a++){
	for (int b=0; b<a; b++){
	  double integral = int2(closed[i],open[a],closed[j],open[b]) - int2(closed[i],open[b],closed[j],open[a]);
	  
	  if (fabs(integral) > epsilon ) {
	    Determinant di = d;
	    di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);
	    if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {

	      {
		int A = (closed[i]), B = closed[j], I= open[a], J = open[b]; 
		double sgn = parity(detArray,norbs,A)*parity(detArray,norbs,I)*parity(detArray,norbs,B)*parity(detArray,norbs,J);
		if (B > J) sgn*=-1 ;
		if (I > J) sgn*=-1 ;
		if (I > B) sgn*=-1 ;
		if (A > J) sgn*=-1 ;
		if (A > B) sgn*=-1 ;
		if (A > I) sgn*=-1 ;
		integral *= sgn;
	      }
	      det_it = dets.find(di);

	      if (additions) {
		if (det_it == dets.end()) dets[di] = integral*ci;
		else det_it->second +=integral*ci;
	      }
	      else 
		if (det_it != dets.end()) det_it->second +=integral*ci;
	    }
	  }
	}
      }
    }
  }
    
  //for (int thrd=0; thrd<omp_get_max_threads(); thrd++)
  //for (int i=0; i<thrdDeterminants[thrd].size(); i++)
  //dets.insert(thrdDeterminants[thrd][i]);
  
  return;
}


void CIPSIbasics::getDeterminants(Determinant& d, double epsilon, double ci, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, vector<int>& irreps, double coreE, double E0, std::map<Determinant, pair<double,double> >& dets, std::vector<Determinant>& Psi0, int oneOrTwo) {

  int norbs = d.norbs;
  int open[norbs], closed[norbs]; char detArray[norbs], diArray[norbs];
  int nclosed = d.getOpenClosed(open, closed);
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);
  
  std::map<Determinant, pair<double,double> >::iterator det_it;
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    if (irreps[closed[i]/2] != irreps[open[a]/2]) continue;
    double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);
    if (fabs(integral) > epsilon ) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {
	det_it = dets.find(di);

	if (oneOrTwo == 0) {
	  if (det_it == dets.end()) dets[di] = make_pair(integral*ci, 0.0);
	  else det_it->second.first +=integral*ci;
	}
	else 
	  if (det_it != dets.end()) det_it->second.second +=integral*ci;
      }
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {

    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>, compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);

    //THERE IS A BUG IN THE CODE WHEN USING HEATBATH INTEGRALS
    if (true && (ints != I2hb.sameSpin.end() || ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>,compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (!(d.getocc(a) || d.getocc(b))) {
	  Determinant di = d;
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    
	  if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {

	    double sgn = 1.0;
	    {
	      int A = (closed[i]), B = closed[j], I= a, J = b; 
	      sgn = parity(detArray,norbs,A)*parity(detArray,norbs,I)*parity(detArray,norbs,B)*parity(detArray,norbs,J);
	      if (B > J) sgn*=-1 ;
	      if (I > J) sgn*=-1 ;
	      if (I > B) sgn*=-1 ;
	      if (A > J) sgn*=-1 ;
	      if (A > B) sgn*=-1 ;
	      if (A > I) sgn*=-1 ;
	    }

	    det_it = dets.find(di);
	    //if (additions) {
	    if (oneOrTwo == 0) {
	      if (det_it == dets.end()) dets[di] = make_pair(it->first*sgn*ci, 0.0);
	      //if (det_it == dets.end()) dets[di] = it->first*ci*sgn;
	      else det_it->second.first +=it->first*ci*sgn;
	    }
	    else 
	      if (det_it != dets.end()) det_it->second.second +=it->first*ci*sgn;

	  }
	}
      }
    }
    else {
      for (int a=0; a<nopen; a++){
	for (int b=0; b<a; b++){
	  double integral = int2(closed[i],open[a],closed[j],open[b]) - int2(closed[i],open[b],closed[j],open[a]);
	  
	  if (fabs(integral) > epsilon ) {
	    Determinant di = d;
	    di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);
	    if (!(binary_search(Psi0.begin(), Psi0.end(), di))) {

	      {
		int A = (closed[i]), B = closed[j], I= open[a], J = open[b]; 
		double sgn = parity(detArray,norbs,A)*parity(detArray,norbs,I)*parity(detArray,norbs,B)*parity(detArray,norbs,J);
		if (B > J) sgn*=-1 ;
		if (I > J) sgn*=-1 ;
		if (I > B) sgn*=-1 ;
		if (A > J) sgn*=-1 ;
		if (A > B) sgn*=-1 ;
		if (A > I) sgn*=-1 ;
		integral *= sgn;
	      }
	      det_it = dets.find(di);

	      if (oneOrTwo == 0) {
		//if (additions) {
		if (det_it == dets.end()) dets[di] = make_pair(integral*ci, 0.0);
		//if (det_it == dets.end()) dets[di] = integral*ci;
		else det_it->second.first +=integral*ci;
	      }
	      else 
		if (det_it != dets.end()) det_it->second.second +=integral*ci;
	    }
	  }
	}
      }
    }
  }
    
  //for (int thrd=0; thrd<omp_get_max_threads(); thrd++)
  //for (int i=0; i<thrdDeterminants[thrd].size(); i++)
  //dets.insert(thrdDeterminants[thrd][i]);
  
  return;
}



  
void CIPSIbasics::getDeterminants(Determinant& d, double epsilon, oneInt& int1, twoInt& int2, twoIntHeatBath& I2hb, double coreE, double E0, std::set<Determinant>& dets) {
  
  int norbs = d.norbs;
  unsigned short open[norbs], closed[norbs]; char detArray[norbs], diArray[norbs];
  int nclosed = d.getOpenClosed(open, closed);
  int nopen = norbs-nclosed;
  d.getRepArray(detArray);
  
  for (int ia=0; ia<nopen*nclosed; ia++){
    int i=ia/nopen, a=ia%nopen;
    double integral = Hij_1Excite(closed[i],open[a],int1,int2, detArray, norbs);
    if (fabs(integral) > epsilon ) {
      Determinant di = d;
      di.setocc(open[a], true); di.setocc(closed[i],false);
      dets.insert(di);
    }
  }
  
  if (fabs(int2.maxEntry) <epsilon) return;

  //#pragma omp parallel for schedule(dynamic)
  for (int ij=0; ij<nclosed*nclosed; ij++) {
    int i=ij/nclosed, j = ij%nclosed;
    if (i<=j) continue;
    int I = closed[i]/2, J = closed[j]/2;
    std::pair<int,int> IJpair(max(I,J), min(I,J));
    std::map<std::pair<int,int>, std::multimap<double, std::pair<int,int>,compAbs > >::iterator ints = closed[i]%2==closed[j]%2 ? I2hb.sameSpin.find(IJpair) : I2hb.oppositeSpin.find(IJpair);
    if (true && (ints != I2hb.sameSpin.end() || ints != I2hb.oppositeSpin.end())) { //we have this pair stored in heat bath integrals
      for (std::multimap<double, std::pair<int,int>, compAbs >::reverse_iterator it=ints->second.rbegin(); it!=ints->second.rend(); it++) {
	if (fabs(it->first) <epsilon) break; //if this is small then all subsequent ones will be small
	int a = 2* it->second.first + closed[i]%2, b= 2*it->second.second+closed[j]%2;
	if (!(d.getocc(a) || d.getocc(b))) {
	  Determinant di = d;
	  di.setocc(a, true), di.setocc(b, true), di.setocc(closed[i],false), di.setocc(closed[j], false);	    
	  dets.insert(di);
	}
      }
    }
    else {
      for (int a=0; a<nopen; a++){
	for (int b=0; b<a; b++){
	  double integral = abs(int2(closed[i],open[a],closed[j],open[b]) - int2(closed[i],open[b],closed[j],open[a]));
	  
	  if (fabs(integral) > epsilon ) {
	    Determinant di = d;
	    di.setocc(open[a], true), di.setocc(open[b], true), di.setocc(closed[i],false), di.setocc(closed[j], false);
	    dets.insert(di);
	  }
	}
      }
    }
  }
    
  //for (int thrd=0; thrd<omp_get_max_threads(); thrd++)
  //for (int i=0; i<thrdDeterminants[thrd].size(); i++)
  //dets.insert(thrdDeterminants[thrd][i]);
  
  return;
}



