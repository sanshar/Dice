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
