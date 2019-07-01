/*
Collatz code for CS 4380 / CS 5351
Copyright (c) 2019 Texas State University. All rights reserved.
Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Author: Martin Burtscher
*/
#include <cstdio>
#include <algorithm>
#include <sys/time.h>
static int collatz(const long range, const long threads)
{
  // compute sequence lengths
  int maxlen = 0;
  #pragma omp parallel for num_threads(threads) default(none) reduction(max:maxlen) SCHED
  for (long i = 1; i <= range; i += 2) {
    long val = i;
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val = val / 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }
    maxlen = std::max(maxlen, len);
  }
  return maxlen;
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.1\n");
  
  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s range numThreads\n", argv[0]); exit(-1);}
  const long range = atol(argv[1]);
  const long threads = atol(argv[2]);
  if (threads < 1){fprintf(stderr, "ERROR: threads must be at least 1\n"); exit(-1);}
  if (range < 3) {fprintf(stderr, "ERROR: range must be at least 3\n"); exit(-1);}
  printf("range bound: %ld\n", range);
  printf("threadcount: %ld\n", threads);
  
  // start time
  timeval start, end;
  gettimeofday(&start, NULL);
  
  // call timed function
  const int maxlen = collatz(range, threads);
  
  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);
  
  // print result
  printf("longest sequence: %d elements\n", maxlen);
  return 0;
}