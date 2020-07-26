#pragma once
#include <string>

#include "CxMemoryStack.h"

struct Kernel {
  virtual void getValueRSpace(double* pFmT, double T, int L, double factor, double rho, double eta, ct::FMemoryStack& Mem) =0;
  virtual inline double getValueKSpace(double T, double factor, double eta)=0;
  virtual std::string getname() = 0;
};

struct CoulombKernel : public Kernel {
  std::string name;
  CoulombKernel()  {name = "coulomb";}
  void getValueRSpace(double* pFmT, double T, int L, double factor, double rho, double eta, ct::FMemoryStack& Mem);
  inline double getValueKSpace(double T, double factor, double eta);
  std::string getname() {return name;}
};

struct KineticKernel : public Kernel {
  std::string name ;
  KineticKernel()  {name = "kinetic" ;}
  void getValueRSpace(double* pFmT, double T, int L, double factor, double rho, double eta, ct::FMemoryStack& Mem);
  inline double getValueKSpace(double T, double factor, double eta);
  std::string getname() {return name;}
};

struct OverlapKernel : public Kernel {
  std::string name;
  OverlapKernel()  {name = "overlap";}
  void getValueRSpace(double* pFmT, double T, int L, double factor, double rho, double eta, ct::FMemoryStack& Mem);
  inline double getValueKSpace(double T, double factor, double eta);
  std::string getname() {return name;}
};

  
