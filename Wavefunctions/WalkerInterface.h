/*
  Developed by Sandeep Sharma with contributions from James E. T. Smith and Adam A. Holmes, 2017
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
#ifndef WalkerInterface_HEADER_H
#define WalkerInterface_HEADER_H

#include <boost/serialization/serialization.hpp>
#include <Eigen/Dense>

class Determinant;
class Wavefunction;

using namespace Eigen;


struct Walker {
  
  virtual void initWalker(const Wavefunction& w) = 0;
  virtual void initWalker(const Wavefunction& w, Determinant &d) = 0;

  virtual void updateWalker(const Wavefunction& w, int ex1, int ex2) = 0;
  //virtual void exciteTo(const Wavefunction& w, Determinant& dcopy) = 0;
  virtual void exciteWalker(const Wavefunction& w, int excite1, int excite2, int norbs) = 0;
  
  virtual double Overlap(const Wavefunction& w) = 0;
  virtual void HamAndOvlp(const Wavefunction& w, double& ovlp, double& ham,
                          workingArray& work, bool fillExcitations = true) = 0;
  virtual void OverlapWithGradient(const Wavefunction& w, double& ovlp, Eigen::VectorXd& v) = 0;
}
#endif
