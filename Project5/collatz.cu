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
#include <cuda.h>
#include <sys/time.h>

static const int ThreadsPerBlock = 512;
//req (4),(3),(2),(8)
static __global__ void collatz(const long range, int* maxlen)
{
  // compute sequence lengths
  //req (7), (5).
  const long idx = threadIdx.x + blockIdx.x * (long)blockDim.x;   long val = idx+1;
  int len = 1;

  if(idx < range)
  while (val != 1) {
     len++;
     if ((val % 2) == 0) {
        val = val / 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }
    if(*maxlen < len) {atomicMax(maxlen, len);}//req (9).

}

static void CheckCuda(){
   cudaError_t e;
   cudaDeviceSynchronize();
   if(cudaSuccess != (e = cudaGetLastError()))
   {fprintf(stderr, "CUDA error %D: %s\n", e, cudaGetErrorString(e)); exit(-1);}
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.1\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s range\n", argv[0]); exit(-1);}
  const long range = atol(argv[1]);
  if (range < 3) {fprintf(stderr, "ERROR: range must be at least 3\n"); exit(-1);}
  printf("range bound: %ld\n", range);

  //allocate mem for deviceMaxlen.
  int* d_maxlen;
  const int size = sizeof(int);
  cudaMalloc((void **)&d_maxlen,size);     //initialize hostMaxlen
  int *h_maxlen = new int;
  *h_maxlen = 0;

  //copy to gpu (Device)
  if (cudaSuccess != cudaMemcpy(d_maxlen, h_maxlen, size, cudaMemcpyHostToDevice)){
     fprintf(stderr, "Copy operation to GPU failed");
     exit(-1);
  }

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // call timed function
  collatz<<<(ThreadsPerBlock + range - 1)/ThreadsPerBlock,ThreadsPerBlock>>>(range, d_maxlen);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);
  CheckCuda();

  //copy data back to host/cpu
  if (cudaSuccess != cudaMemcpy(h_maxlen, d_maxlen, size, cudaMemcpyDeviceToHost))
  {fprintf(stderr, "copy from gpu to cpu failed!\n"); exit(-1);}

  // print result
  printf("longest sequence: %d elements\n", *h_maxlen);

  //deleting memory
  delete h_maxlen;
  cudaFree(d_maxlen);
  return 0;
}
