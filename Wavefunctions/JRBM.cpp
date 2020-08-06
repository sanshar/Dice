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
#include "JRBM.h"
#include "Determinants.h"
#include <boost/container/static_vector.hpp>
#include <fstream>
#include "input.h"

using namespace Eigen;

JRBM::JRBM () {};

double JRBM::Overlap(const Determinant &d) const
{
  return jastrow.Overlap(d) * rbm.Overlap(d);
}


double JRBM::OverlapRatio (const Determinant &d1, const Determinant &d2) const {
  return Overlap(d1)/Overlap(d2);
}


double JRBM::OverlapRatio(int i, int a, const Determinant &dcopy, const Determinant &d) const
{
  return OverlapRatio(dcopy, d);
}

double JRBM::OverlapRatio(int i, int j, int a, int b, const Determinant &dcopy, const Determinant &d) const
{
  return OverlapRatio(dcopy, d);
}


void JRBM::OverlapWithGradient(const Determinant& d, 
                              Eigen::VectorBlock<VectorXd>& grad,
                              const double& ovlp) const {
  VectorXd gradVec = VectorXd::Zero(grad.size());
  Eigen::VectorBlock<VectorXd> gradhead = gradVec.head(jastrow.getNumVariables());
  jastrow.OverlapWithGradient(d, gradhead, ovlp);
  Eigen::VectorBlock<VectorXd> gradtail = gradVec.tail(rbm.getNumVariables());
  rbm.OverlapWithGradient(d, gradtail, ovlp);
  grad = gradVec;
}

long JRBM::getNumVariables() const
{
  return jastrow.getNumVariables() + rbm.getNumVariables();
}


void JRBM::getVariables(Eigen::VectorBlock<VectorXd> &v) const
{
  VectorXd vVec = VectorXd::Zero(v.size());
  Eigen::VectorBlock<VectorXd> vhead = vVec.head(jastrow.getNumVariables());
  jastrow.getVariables(vhead);
  Eigen::VectorBlock<VectorXd> vtail = vVec.tail(rbm.getNumVariables());
  rbm.getVariables(vtail);
  v = vVec;
}

void JRBM::updateVariables(const Eigen::VectorBlock<VectorXd> &v)
{
  VectorXd vVec = v;
  Eigen::VectorBlock<VectorXd> vhead = vVec.head(jastrow.getNumVariables());
  jastrow.updateVariables(vhead);
  Eigen::VectorBlock<VectorXd> vtail = vVec.tail(rbm.getNumVariables());
  rbm.updateVariables(vtail);
}

void JRBM::printVariables() const
{
  jastrow.printVariables();
  rbm.printVariables();
}
