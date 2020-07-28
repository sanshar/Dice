#pragma once

#include "CxMemoryStack.h"

enum KernelID {coulombKernel, kineticKernel, overlapKernel}; 

struct Kernel {
  virtual void getValueRSpace(double* pFmT, double T, int L, double factor, double rho, double eta, ct::FMemoryStack& Mem) =0;
  virtual inline double getValueKSpace(double T, double factor, double eta)=0;
  virtual int getname() = 0;
};

struct CoulombKernel : public Kernel {
  int name;
  CoulombKernel()  {name = coulombKernel;}
  void getValueRSpace(double* pFmT, double T, int L, double factor, double rho, double eta, ct::FMemoryStack& Mem);
  inline double getValueKSpace(double T, double factor, double eta);
  int getname() {return name;}
};

struct KineticKernel : public Kernel {
  int name ;
  KineticKernel()  {name = kineticKernel ;}
  void getValueRSpace(double* pFmT, double T, int L, double factor, double rho, double eta, ct::FMemoryStack& Mem);
  inline double getValueKSpace(double T, double factor, double eta);
  int getname() {return name;}
};

struct OverlapKernel : public Kernel {
  int name;
  OverlapKernel()  {name = overlapKernel;}
  void getValueRSpace(double* pFmT, double T, int L, double factor, double rho, double eta, ct::FMemoryStack& Mem);
  inline double getValueKSpace(double T, double factor, double eta);
  int getname() {return name;}
};

  
