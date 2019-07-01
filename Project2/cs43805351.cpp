#ifndef CS43805351
#define CS43805351

#include <stdio.h>
#include <assert.h>
#include <math.h>

static void writeBMP(const int x, const int y, const unsigned char* const bmp, const char* const name)
{
  const unsigned char bmphdr[54] = {66, 77, 255, 255, 255, 255, 0, 0, 0, 0, 54, 4, 0, 0, 40, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 1, 0, 8, 0, 0, 0, 0, 0, 255, 255, 255, 255, 196, 14, 0, 0, 196
, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  unsigned char hdr[1078];
  int i, j, c, xcorr, diff;
  FILE* f;

  xcorr = (x + 3) >> 2 << 2;  // BMPs have to be a multiple of 4 pixels wide
  diff = xcorr - x;

  for (i = 0; i < 54; i++) hdr[i] = bmphdr[i];
  *((int*)(&hdr[18])) = xcorr;
  *((int*)(&hdr[22])) = y;
  *((int*)(&hdr[34])) = xcorr * y;
  *((int*)(&hdr[2])) = xcorr * y + 1078;
  for (i = 0; i < 256; i++) {
    j = i * 4 + 54;
    hdr[j+0] = 128 + sin((i + 85) * 0.0245436926) * 127;  // blue
    hdr[j+1] = 128 + sin((i + 0) * 0.0245436926) * 127;  // green
    hdr[j+2] = 128 + sin((i + 171) * 0.0245436926) * 127;  // red
    hdr[j+3] = 0;  // dummy
  }

  f = fopen(name, "wb");  assert(f != NULL);
  c = fwrite(hdr, 1, 1078, f);  assert(c == 1078);
  if (diff == 0) {
    c = fwrite(bmp, 1, x * y, f);  assert(c == x * y);
  } else {
    *((int*)(&hdr[0])) = 0;  // need up to three zero bytes
    for (j = 0; j < y; j++) {
      c = fwrite(&bmp[j * x], 1, x, f);  assert(c == x);
      c = fwrite(hdr, 1, diff, f);  assert(c == diff);
    }
  }
  fclose(f);
}

#endif