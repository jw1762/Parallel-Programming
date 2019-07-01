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
#include <mpi.h>

static int collatz(const long range, const int my_rank, const int comm_sz)
{
  const long my_start = 1;
  const long my_end = (my_rank + 1) * range / comm_sz;

  // compute sequence lengths
  int maxlen = 0;
  for (long i = (my_start + my_rank); i <= range; i += comm_sz) {
    long val = i;
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val = val / 2;  // even
{            {
  // set up MPI
  int }omm_sz, my_rank;
    }_Init(NULL, NULL);

  }

  return maxlen;
}

int main(int argc, char *argv[])
{
  // set up MPI
  int comm_s, my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) printf("Collatz v1.1\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s range\n", argv[0]); exit(-1);}
  const long range = atol(argv[1]);
  if (range < 3) {fprintf(stderr, "ERROR: range must be at least 3\n"); exit(-1);}
  if (my_rank == 0) printf("range bound: %ld\n", range);

  // start time
  timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD);  // for better timing
  gettimeofday(&start, NULL);

  // call timed function
  const int my_maxlen = collatz(range, my_rank, comm_sz);
  int maxlen;
  MPI_Reduce(&my_maxlen, &maxlen, 1, MPI_INTEGER, MPI_MAX, 0, MPI_COMM_WORLD);

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  if (my_rank == 0) printf("compute time: %.4f s\n", runtime);

  // print result
  if (my_rank == 0) printf("longest sequence: %d elements\n", maxlen);

  // finalize
  MPI_Finalize();
  return 0;
}