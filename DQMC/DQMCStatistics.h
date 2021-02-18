#ifndef DQMCStatistics_HEADER_H
#define DQMCStatistics_HEADER_H
#include <utility>
#include <vector>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <Eigen/Dense>

// holds numerator and denominator samples from direct sampling in dqmc
class DQMCStatistics {
 private:
   friend class boost::serialization::access;
   template<class Archive> 
     void serialize(Archive & ar, const unsigned int version) {
       ar & sampleSize
          & nSamples
          & numSamples
          & denomSamples
          & numMean
          & denomMean
          & denomAbsMean
          & num2Mean
          & denom2Mean
          & num_denomMean
          & errorTargets
          & converged
          & convergedE
          & convergedDev
          & convergedDev2
          & convergedPhase;
     }
  
 public:

    int sampleSize;                                        // samples are taken at multiple points in a sweep
    size_t nSamples;                                       // number of samples
    std::vector<Eigen::ArrayXcd> numSamples, denomSamples; // the samples
    Eigen::ArrayXcd numMean, denomMean;                    // running averages
    Eigen::ArrayXd denomAbsMean;                           // running averages
    Eigen::ArrayXcd num2Mean, denom2Mean, num_denomMean;   // running averages
    std::vector<double> errorTargets;
    std::vector<int> converged;
    Eigen::ArrayXcd convergedE, convergedPhase;
    Eigen::ArrayXd convergedDev, convergedDev2;

    // constructor
    DQMCStatistics(int pSampleSize);

    // store samples and update running averages
    void addSamples(Eigen::ArrayXcd& numSample, Eigen::ArrayXcd& denomSample);
    
    // get the current number of samples
    size_t getNumSamples();
    
    // calculates error by blocking data
    // use after gathering data across processes for better estimates
    void calcError(Eigen::ArrayXd& error, Eigen::ArrayXd& error2, Eigen::ArrayXcd& bias);
    
    // gather data from all the processes and print quantities
    // iTime used only for printing
    // delta is used for energy extrapolation
    void gatherAndPrintStatistics(Eigen::ArrayXd iTime, std::complex<double> delta = std::complex<double>(0., 0.));

    // if all energies are converged
    bool isConverged();

    // prints running averages from proc 0
    void printStatistics();

    // write samples to disk
    void writeSamples();
    
    //// write bkp file
    //void saveStats() 
    //{
    //  char file[5000];
    //  std::sprintf (file, "dqmcStats_%d.bkp", commrank);
    //  std::ofstream outfs(file, std::ios::binary);
    //  boost::archive::binary_oarchive save(outfs);
    //  save << *this;
    //  outfs.close();
    //};
    //
    //// read saved bkp file
    //void loadStats()
    //{
    //  char file[5000];
    //  std::sprintf (file, "dqmcStats_%d.bkp", commrank);
    //  std::ifstream infs(file, std::ios::binary);
    //  boost::archive::binary_iarchive load(infs);
    //  load >> *this;
    //  infs.close();
    //
    //  // if restarting do at least one sweep
    //  for (int i = 0; i < sampleSize; i++) converged[i] = -1;  
    //};
}; 

#endif
