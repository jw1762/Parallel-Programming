/*N-body code for CS 4380 / CS 5351

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

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <sys/time.h>
#include "cs43805351.h"
#include <pthread.h>//(2).

static int nbodies;
static int genimages;
static int timesteps;
static int step;
static int numThreads;//numThreads
static long thread;//thread ID

static long my_rank;
static int beg;
static int end;

pthread_t* threads = NULL; //array ptr of thread ID

static const int WIDTH = 512;
static const float dthf = 0.025f * 0.5f;

struct Data {              // mass, 3D position, 3D velocity, 3D acceleration
  float mass, posx, posy, posz, velx, vely, velz, accx, accy, accz;
};

static void outputBMP(const int nbodies, const Data* const data, const int step)
{
  unsigned char* bmp = new unsigned char [WIDTH * WIDTH];
  for (int i = 0; i < WIDTH * WIDTH; i++) bmp[i] = 0;

  for (int i = 0; i < nbodies; i++) {
    const float fz = data[i].posz + 3.0f;
    if (fz > 0) {
      const float fx = data[i].posx;
      const float fy = data[i].posy;
      const float dsqr = fx * fx + fy * fy + fz * fz;
      const int x = atanf(fx / fz) * (WIDTH / 2) + (0.5f + WIDTH / 2);
      const int y = atanf(fy / fz) * (WIDTH / 2) + (0.5f + WIDTH / 2);
      int c = 140 - dsqr * 4.0f;
      if (c < 100) c = 100;
      if ((0 <= x) && (x < WIDTH) && (0 <= y) && (y < WIDTH)) {
        if (c > bmp[x + y * WIDTH]) bmp[x + y * WIDTH] = c;
      }
    }
  }

  char name[32];
  sprintf(name, "nbody%d.bmp", step + 1000);
  writeBMP(WIDTH, WIDTH, bmp, name);

  delete [] bmp;
}

/******************************************************************************/
/*** generate input (based on SPLASH2) ****************************************/
/******************************************************************************/

static const int MASK = 0x7FFFFFFF;
static int randx = 1;

static double drnd()
{
  const int lastrand = randx;
  randx = (1103515245 * randx + 12345) & MASK;
  return lastrand / 2147483648.0;
}

static void generateInput(const int nbodies, Data* const data)  {
  const double rsc = 0.5890486225481;
  const double vsc = sqrt(1.0 / rsc);

  for (int i = 0; i < nbodies; i++) {
    data[i].mass = 1.0 / nbodies;

    const double r = 1.0 / sqrt(pow(drnd() * 0.999, -2.0 / 3.0) - 1);
    double x, y, z, sq;
    do {
      x = drnd() * 2.0 - 1.0;
      y = drnd() * 2.0 - 1.0;
      z = drnd() * 2.0 - 1.0;
      sq = x * x + y * y + z * z;
    } while (sq > 1.0);
    double scale = rsc * r / sqrt(sq);
    data[i].posx = x * scale;
    data[i].posy = y * scale;
    data[i].posz = z * scale;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x * x * pow(1 - x * x, 3.5));
    const double v = x * sqrt(2.0 / sqrt(1 + r * r));
    do {
      x = drnd() * 2.0 - 1.0;
      y = drnd() * 2.0 - 1.0;
      z = drnd() * 2.0 - 1.0;
      sq = x * x + y * y + z * z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    data[i].velx = x * scale;
    data[i].vely = y * scale;
    data[i].velz = z * scale;
  }

  for (int i = 0; i < nbodies; i++) {
    data[i].accx = 0;
    data[i].accy = 0;
    data[i].accz = 0;
  }
}

/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

static void* calculateForce(void* rank){
  const float epssq = 0.05f * 0.05f;

  my_rank = (long)rank;
  beg = my_rank * nbodies/numThreads;
  end = (my_rank + 1) * nbodies/numThreads;

  for (int i = beg; i < end; i++) {
    const float px = data[i].posx;
    const float py = data[i].posy;
    const float pz = data[i].posz;

    float ax = 0;
    float ay = 0;
    float az = 0;

    for (int j = 0; j < nbodies; j++) {
      const float dx = data[j].posx - px;
      const float dy = data[j].posy - py;
      const float dz = data[j].posz - pz;
      float tmp = dx * dx + dy * dy + dz * dz;
      tmp = 1.0f / sqrtf(tmp + epssq);
      tmp = data[j].mass * tmp * tmp * tmp;
      ax += dx * tmp;
      ay += dy * tmp;
      az += dz * tmp;
    }

    if (step > 0) {
      data[i].velx += (ax - data[i].accx) * dthf;
      data[i].vely += (ay - data[i].accy) * dthf;
      data[i].velz += (az - data[i].accz) * dthf;
    }

    data[i].accx = ax;
    data[i].accy = ay;
    data[i].accz = az;
  }
  return NULL;
}

/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/

static void* integrate(void* rank)
{
  my_rank = (long)rank;
  beg = my_rank * nbodies/numThreads; end = (my_rank+1) * nbodies/numThreads;

  const float dtime = dthf + dthf;
  for (int i = beg; i < end; i++) {
    const float dvelx = data[i].accx * dthf;
    const float dvely = data[i].accy * dthf;
    const float dvelz = data[i].accz * dthf;

    const float velhx = data[i].velx + dvelx;
    const float velhy = data[i].vely + dvely;
    const float velhz = data[i].velz + dvelz;

    data[i].posx += velhx * dtime;
    data[i].posy += velhy * dtime;
    data[i].posz += velhz * dtime;

    data[i].velx = velhx + dvelx;
    data[i].vely = velhy + dvely;
    data[i].velz = velhz + dvelz;
  }
  return NULL;
}

static void* forceCalc()
{
   threads = new pthread_t[numThreads-1];

   for (thread = 0; thread < numThreads -1; thread++)
   {
      pthread_create(&threads[thread], NULL, calculateForce, (void*)(thread+1));

   }
   calculateForce((void*)0);
   for (thread = 0; thread < numThreads; thread++) { pthread_join(threads[thread], NULL); }

   delete [] threads;
   return NULL;
}

static void* forceInteg()
{
   threads = new pthread_t[numThreads-1];
   for (thread = 0; thread < numThreads - 1; thread++)
   {pthread_create(&threads[thread], NULL, integrate, (void*)(thread+1));}
   integrate((void*)0);
   for (thread = 0; thread < numThreads; thread++)
   {pthread_join(threads[thread], NULL);}
   delete [] threads;
   return NULL;
   }

int main(int argc, char *argv[])
{
  printf("N-body v1.1\n");

  // check command line
  if (argc != 5) {fprintf(stderr, "USAGE: %s number_of_bodies number_of_timesteps generate_images num_of_threads\n", argv[0]); exit(-1);}
  nbodies = atoi(argv[1]);
  if (nbodies < 10) {fprintf(stderr, "ERROR: number_of_bodies must be at least 10\n"); exit(-1);}
  timesteps = atoi(argv[2]);
  if (timesteps < 1) {fprintf(stderr, "ERROR: number_of_timesteps must be at least 1\n"); exit(-1);}
  genimages = atoi(argv[3]);
  if ((genimages != 0) && (genimages != 1)) {fprintf(stderr, "ERROR: generate_images must be either 0 or 1\n"); exit(-1);}
  numThreads = atoi(argv[4]);
  if (numThreads < 1) {fprintf(stderr, "ERROR: num_of_threads must be at least 1\n"); exit(-1);}//(4)

  printf("bodies: %d\n", nbodies);
  printf("time steps: %d\n", timesteps);
  printf("images: %s\n", genimages ? "yes" : "no");
  printf("threads: %d\n", numThreads);//(5).

  // allocate and initialize data
  Data* data = new Data[nbodies];
  generateInput(nbodies, data);

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // compute result for each time step
  const float dthf = 0.025f * 0.5f;
  for (int step = 0; step < timesteps; step++) {
    forceCalc();
    forceInteg();
    // write result to BMP file
    if (genimages) outputBMP(nbodies, data, step);
  }

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  delete [] data;
  return 0;
}