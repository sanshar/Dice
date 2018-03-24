/*
  Developed by Sandeep Sharma
  Copyright (c) 2017, Sandeep Sharma
  
  This file is part of DICE.
  
  This program is free software: you can redistribute it and/or modify it under the terms
  of the GNU General Public License as published by the Free Software Foundation, 
  either version 3 of the License, or (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  See the GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License along with this program. 
  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef Wfn_HEADER_H
#define Wfn_HEADER_H
#include <vector>
#include "MoDeterminants.h"
#include "Determinants.h"
#include "CPS.h"
#include <boost/serialization/vector.hpp>

class oneInt;
class twoInt;
class Walker;

class Wfn {
 public:
  virtual double Overlap(Determinant&) =0;
  virtual void HamAndOvlp(Determinant& ,
			  double& ovlp, double& ham,
			  oneInt& I1, twoInt& I2, double& coreE) =0;
  virtual void HamAndOvlp(Walker& ,
		  double& ovlp, double& ham,
		  oneInt& I1, twoInt& I2, double& coreE)=0;
  virtual void HamAndOvlpGradient(Walker& ,
				  double& ovlp, double& ham, Eigen::VectorXd& grad, double& scale,
				  double& E0, oneInt& I1, twoInt& I2, double& coreE)=0;
  virtual void OverlapWithGradient(Determinant&, 
				   double& factor,
				   Eigen::VectorXd& grad)=0;
  virtual void OverlapWithGradient(Walker&, 
				   double& factor,
				   Eigen::VectorXd& grad)=0;
  virtual void printVariables() =0;
  virtual void getDetMatrix(Determinant&, Eigen::MatrixXd& alpha, Eigen::MatrixXd& beta)=0;

};


class CPSSlater : public Wfn {
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & cpsArray & det;
  }
 public:
  std::vector<CPS> cpsArray;
  MoDeterminant det;

  CPSSlater( std::vector<CPS>& pcpsArray, MoDeterminant& pdet) : cpsArray(pcpsArray), det(pdet) {};

  
  void HamAndOvlpGradient(Walker& ,
			  double& ovlp, double& ham, Eigen::VectorXd& grad, double& scale,
			  double& E0, oneInt& I1, twoInt& I2, double& coreE);
  void OverlapWithGradient(Determinant&, 
			   double& factor,
			   Eigen::VectorXd& grad);
  void OverlapWithGradient(Walker&, 
			   double& factor,
			   Eigen::VectorXd& grad);

  double Overlap(Determinant&);
  void HamAndOvlp(Determinant& ,
		  double& ovlp, double& ham,
		  oneInt& I1, twoInt& I2, double& coreE);
  void HamAndOvlp(Walker& ,
		  double& ovlp, double& ham,
		  oneInt& I1, twoInt& I2, double& coreE);

  void getVariables(Eigen::VectorXd& v);
  long getNumVariables();
  void updateVariables(Eigen::VectorXd& dv);
  void incrementVariables(Eigen::VectorXd& dv);
  void printVariables();
  void writeWave();
  void readWave();
  void getDetMatrix(Determinant&, Eigen::MatrixXd& alpha, Eigen::MatrixXd& beta);
};

#endif
