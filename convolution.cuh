#ifndef _CONVOLUTION_CUH_
#define _CONVOLUTION_CUH_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <helper_cuda.h>
#include <helper_math.h>

__global__ void d_slowConvolution(unsigned int *d_img, unsigned int *d_result, float *d_kernel, int width, int height, int radius);
double convolution(unsigned int *d_img, unsigned int *d_result, float *d_kernel, int width, int height, int radius);

#endif // #ifndef _CONVOLUTION_CUH_
