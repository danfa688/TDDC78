#include "thresfilter.h"

void thresfilter(const int xsize, const int ysize, pixel* src, const int sum){
#define uint unsigned int 

  uint i, psum, nump;

  nump = xsize * ysize;

  for(i = 0; i < nump; i++) {
    psum = (uint)src[i].r + (uint)src[i].g + (uint)src[i].b;
    if(sum > psum) {
      src[i].r = src[i].g = src[i].b = 0;
    }
    else {
      src[i].r = src[i].g = src[i].b = 255;
    }
  }
}

uint localthreshold(const int xsize, const int ysize, pixel* src){
#define uint unsigned int 

  uint sum, i, nump;

  nump = xsize * ysize;

  for(i = 0, sum = 0; i < nump; i++) {
    sum += (uint)src[i].r + (uint)src[i].g + (uint)src[i].b;
  }

  sum /= nump;
  return sum;
}
