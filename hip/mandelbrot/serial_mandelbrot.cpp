// Note: Most of the code comes from the MacResearch OpenCL podcast

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>

#include "CImg.h"

/* Mandlebrot rendering function
   :inputs: width and height of domain, max_iterations
   :ouputs: 8biti unsigned character array containing mandlebrot image
*/
void render(unsigned char out[], const int width, const int height, const int max_iter) {

  float x_origin, y_origin, xtemp, x, y;
  int iteration, index;
  for(int i = 0; i < width; i++) {
      for(int j = 0; j < height; j++) {
          index = width * j + i;
          iteration = 0
          x = 0.0f;
          y = 0.0f;
          x_origin = ((float) i / width) * 3.25f - 2.0f;
          y_origin = ((float) j / width) * 2.5f - 1.25f;
          while (x*x + y*y <= 4 && iteration < max_iter) {
              xtemp = x*x - y*y + x_origin;
              y = 2*x*y + y_origin;
              x = xtemp;
              iteration++;
          }
          if(iteration==max_iter) {
              out[index] = 0;
          }
          else {
              out[index] = iteration;
          }
          
      }
  }
}
/*
	Conditional loop in the middle -> threads will finish at different times
	Thread divergence the program as fast as the slowest thread.   
*/


/* Host function for generating the mandlebrot image
   :inputs: width and height of domain, and max_iterations for escape
   :outputs: none
   writes a bmp image to disc
*/
void mandelbrot(const int width, const int height, const int max_iter)
{
  unsigned char *image;
  image = new unsigned char[width*height];
  
  render(image, width, height, max_iter);

  // Now write the file
  cimg_library::CImg<unsigned char> img2(host_image, width, height);
  img2.save("output.bmp");

  delete host_image;
}


/*main function */ 
int main() 
{
  int N[] = {1024, 2048, 4096, 8192};
  for(int i = 0; i < 4; ++i) {
    mandelbrot(N[i], N[i], 256);
  }
  // mandelbrot(1024, 1024, 256);
  return 0;
}
