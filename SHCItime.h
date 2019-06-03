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
#ifndef TIME_H_
#define TIME_H_

#include <chrono>
#include <string>

class Time {
 public:
  static void print_time(std::string msg) {
    using namespace std::chrono;
    const auto now = std::chrono::high_resolution_clock::now();
    static auto start = now;
    static auto last = now;
    const double tot = duration_cast<duration<double>>(now - start).count();
    const double diff = duration_cast<duration<double>>(now - last).count();
    printf("TOT: %.3f s. DIFF: %.3f s. MSG: %s\n", tot, diff, msg.c_str());
    fflush(stdout);
    last = now;
  }
};


#endif
