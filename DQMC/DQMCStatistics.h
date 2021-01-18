#ifndef DQMCStatistics_HEADER_H
#define DQMCStatistics_HEADER_H
#include <utility>
#include <vector>
#include <boost/serialization/serialization.hpp>
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
          & converged
          & convergedE
          & convergedDev;
     }
  
 public:

    int sampleSize;                                        // samples are taken at multiple points in a sweep
    size_t nSamples;                                       // number of samples
    std::vector<Eigen::ArrayXcd> numSamples, denomSamples; // the samples
    Eigen::ArrayXcd numMean, denomMean;                    // running averages
    Eigen::ArrayXd denomAbsMean;                           // running averages
    Eigen::ArrayXcd num2Mean, denom2Mean, num_denomMean;   // running averages
    std::vector<int> converged;
    Eigen::ArrayXcd convergedE;
    Eigen::ArrayXd convergedDev;

    // constructor
    DQMCStatistics(int pSampleSize);

    // store samples and update running averages
    void addSamples(Eigen::ArrayXcd& numSample, Eigen::ArrayXcd& denomSample);
    
    // calculates error by blocking data
    // use after gathering data across processes for better estimates
    void calcError(Eigen::ArrayXd& error, Eigen::ArrayXd& error2);
    
    // gather data from all the processes and print quantities
    // to be used at the end of a calculation
    // iTime used only for printing
    void gatherAndPrintStatistics(Eigen::ArrayXd iTime);

    // prints running averages from proc 0
    void printStatistics();

    // write samples to disk
    void writeSamples();
}; 

#endif
