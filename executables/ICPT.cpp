/*                                                                                                                                                                          
Developed by Sandeep Sharma and Gerald Knizia, 2016                                                                                                                     
Copyright (c) 2016, Sandeep Sharma
*/
#include "CxNumpyArray.h"
#include "CxMemoryStack.h"
#include "BlockContract.h"
#include "CxDefs.h"
#include "icpt.h"
#include "mkl.h"
#include "mkl_cblas.h"
using ct::TArray;
using ct::FMemoryStack;
using boost::format;


int main(int argc, char const *argv[])
{
   // what we need:
   //   - CI order, nElec, Ms2  (-> nOccA, nOccB)
   //   - tensor declarations
   //   - equations (read from .inl files)
   //   - E0/Fock/Int2e (take from numpy arrays).
   //     Assume input is already in MO basis.
   //   - something to split up tensors (H) and combine them (RDM)
   //     regarding universal indices

  mkl_set_num_threads(numthrds);

   FJobContext
      Job;
   char const
      *pInp = "CoMinAo.args";
   std::string
      MethodOverride;
   for (int iArg = 1; iArg < argc; ++ iArg) {
      if (0 == strcmp(argv[iArg], "-m")) {
         ++ iArg;
         if (iArg < argc)
            MethodOverride = std::string(argv[iArg]);
         continue;
      }
      pInp = argv[iArg];
   }

   Job.ReadInputFile(pInp);
   if (!MethodOverride.empty())
      Job.MethodName = MethodOverride;

   ct::FMemoryStack2 Mem[numthrds];
   size_t mb = 1048576;
   Mem[0].Create(Job.WorkSpaceMb*mb);

   if (Job.MethodName == "MRLCC") {
     std::string methods[] = {"MRLCC_CCVV", "MRLCC_ACVV", "MRLCC_AAVV", "MRLCC_CCAA", "MRLCC_CCAV", "MRLCC_CAAV"};

     for (int i=0; i<6; i++) {
       Job.MethodName = methods[i];
       Job.Run(Mem);
       Job.DeleteData(Mem[0]);
    }
   }
   else if (Job.MethodName == "NEVPT2") {
     //std::string methods[] = {"NEVPT2_CCVV", "NEVPT2_ACVV", "NEVPT2_AAVV", "NEVPT2_CCAA", "NEVPT2_CCAV", "NEVPT2_CAAV"};
     std::string methods[] = {"NEVPT2_CCVV", "NEVPT2_ACVV", "NEVPT2_CCAV"};

     for (int i=0; i<3; i++) {
       Job.MethodName = methods[i];
       Job.Run(Mem);
       Job.DeleteData(Mem[0]);
    }
   }
   else
     Job.Run(Mem);
}
