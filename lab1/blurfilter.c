/*
  File: blurfilter.c

  Implementation of blurfilter function.
    
 */
#include <stdio.h>
#include "blurfilter.h"
#include "ppmio.h"
#include <stdlib.h>


pixel* pix(pixel* image, const int xx, const int yy, const int xsize, const int datalength)
{
  register int off = xsize*yy + xx;

#ifdef DBG
  if(off >= datalength*xsize) {
    fprintf(stderr, "\n Terribly wrong: %d %d %d\n",xx,yy,xsize);
  }
#endif
  return (image + off);
}

void blurfilter(const int xsize, const int ysize, pixel* src, const int radius, const double *w, const int ydiff, const int ylength, const int datalength){
  int x,y,x2,y2, wi;
  double r,g,b,n, wc;
    pixel* dst;
    dst = malloc(datalength*xsize*sizeof(*dst));
  for (y=0; y<datalength; y++) {
    for (x=0; x<xsize; x++) {
      r = w[0] * pix(src, x, y, xsize, datalength)->r;
      g = w[0] * pix(src, x, y, xsize, datalength)->g;
      b = w[0] * pix(src, x, y, xsize, datalength)->b;
      n = w[0];
      for ( wi=1; wi <= radius; wi++) {
		wc = w[wi];
		x2 = x - wi;
		if(x2 >= 0) {
		  r += wc * pix(src, x2, y, xsize, datalength)->r;
		  g += wc * pix(src, x2, y, xsize, datalength)->g;
		  b += wc * pix(src, x2, y, xsize, datalength)->b;
		  n += wc;
		}
		x2 = x + wi;
		if(x2 < xsize) {
		  r += wc * pix(src, x2, y, xsize, datalength)->r;
		  g += wc * pix(src, x2, y, xsize, datalength)->g;
		  b += wc * pix(src, x2, y, xsize, datalength)->b;
		  n += wc;
		}
	  }
	  pix(dst,x,y, xsize, datalength)->r = r/n;
	  pix(dst,x,y, xsize, datalength)->g = g/n;
	  pix(dst,x,y, xsize, datalength)->b = b/n;
    }
  }

  for (y=ydiff; y<ylength+ydiff; y++) {
    for (x=0; x<xsize; x++) {
      r = w[0] * pix(dst, x, y, xsize, datalength)->r;
      g = w[0] * pix(dst, x, y, xsize, datalength)->g;
      b = w[0] * pix(dst, x, y, xsize, datalength)->b;
      n = w[0];
      for ( wi=1; wi <= radius; wi++) {
		wc = w[wi];
		y2 = y - wi;
		if(y2 >= 0) {
		  r += wc * pix(dst, x, y2, xsize, datalength)->r;
		  g += wc * pix(dst, x, y2, xsize, datalength)->g;
		  b += wc * pix(dst, x, y2, xsize, datalength)->b;
		  n += wc;
		}
		y2 = y + wi;
		if(y2 < datalength) {
		  r += wc * pix(dst, x, y2, xsize, datalength)->r;
		  g += wc * pix(dst, x, y2, xsize, datalength)->g;
		  b += wc * pix(dst, x, y2, xsize, datalength)->b;
		  n += wc;
		}
      }
      pix(src,x,y, xsize, datalength)->r = r/n;
      pix(src,x,y, xsize, datalength)->g = g/n;
      pix(src,x,y, xsize, datalength)->b = b/n;
    }
  }
}



