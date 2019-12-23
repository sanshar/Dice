/* This program is not yet in a release condition and should not be distributed.
 * Author: Gerald Knizia, Aug. 2013.
 */

// include OpenMP header if available or define inline dummy functions
// for OMP primitives if not.
#ifndef OPENMP_PROXY_H
#define OPENMP_PROXY_H

#ifdef _OPENMP
   #include <omp.h>
#else
   inline int omp_get_thread_num() { return 0; } // current thread id
   inline void omp_set_num_threads(int) {};
   inline int omp_get_max_threads() { return 1; } // total number of threads supposed to be running.

   struct omp_lock_t {};
   inline void omp_destroy_lock(omp_lock_t *){};
   inline void omp_init_lock(omp_lock_t *){};
   inline void omp_set_lock(omp_lock_t *){};
#endif

#endif // OPENMP_PROXY_H
