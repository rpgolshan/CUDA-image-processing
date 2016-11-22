#ifndef _CONVOLUTION_CUH_
#define _CONVOLUTION_CUH_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <helper_cuda.h>
#include <helper_math.h>

double convolution(unsigned int *d_img, unsigned int *d_result, int *d_kernel, int width, int height, int radius, int type, int weight, int iterations);

#endif // #ifndef _CONVOLUTION_CUH_
