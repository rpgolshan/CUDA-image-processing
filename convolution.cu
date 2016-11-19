#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "convolution.cuh"

// 2D float texture
//texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;


/* 
 * Converts a uint to a uint3, seperating RGB
 * colors by every byte.
 * Most significat to least significant:
 * Red, Green, Blue
 */
__device__ uint3 d_uintToRGB(unsigned int orig)
{
    uint3 rgb;
    rgb = make_uint3(orig & 0xff, (orig>>8)&0xff, (orig>>16)&0xff);
    return rgb;
}

/*
 * Converts a uint3 to an unsigned int
 * Assumes each vector member correspond to RGB colors
 * Truncates rgb colors bigger than 1 byte
 */
__device__ unsigned int d_rgbToUint(uint3 rgb)
{
    unsigned int total;
    if (rgb.x > 0xff) rgb.x = 0xff;
    if (rgb.y > 0xff) rgb.x = 0xff;
    if (rgb.z > 0xff) rgb.x = 0xff;

    total = (rgb.x & 0xff) + ((rgb.y & 0xff) << 8) + ((rgb.z & 0xff) << 16);
    return total;
}

/* The most basic convolution method in parallel
 * Does not take advantage of memory optimizations with a GPU
 * Can be used with any (square) kernel filter
 * SLOW
 * Each output pixel does radius^2 multiplications 
 * T = O(radius^2)
 * W = O(radius^2 * width * height)
 */
__global__ void d_slowConvolution(unsigned int *d_img, unsigned int *d_result, float *d_kernel, int width, int height, int radius)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int loc = x + (blockIdx.y*blockDim.y)*width + threadIdx.y*width;
    uint3 accumulation = make_uint3(0,0,0);
    uint3 value;
    float weight = 0.0f;

    if (x >= width || y >= height) return;
    assert(x < width);
    assert(y < height);
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            if ((x + i < 0) || //left side out of bounds
                (x + i >= width) || //right side OoB
                (y + j < 0) || //top OoB
                (y + j >= height)) //bot OoB
                value = make_uint3(0,0,0);
            else { 
                value = d_uintToRGB(d_img[loc + i + j * width]);
            }
            float temp = d_kernel[i + radius +  (j+radius)*(radius*2 + 1)];
            value.x *= temp;
            value.y *= temp;
            value.z *= temp;
            weight += temp;
            accumulation += value;
        }
    }
    if (radius == 0) //i.e. original image
        d_result[loc] = d_img[loc];
    else  {
        accumulation.x =  accumulation.x/weight;
        accumulation.y =  accumulation.y/weight;
        accumulation.z =  accumulation.z/weight;
        d_result[loc] = d_rgbToUint(accumulation);
    }
}

double convolution(unsigned int *d_img, unsigned int *d_result, float *d_kernel, int width, int height,
                 int radius)
{
    // sync host and start computation timer_kernel
    checkCudaErrors(cudaDeviceSynchronize());
//    checkCudaErrors(cudaBindTextureToArray(tex, d_array));

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(width / threadsPerBlock.x, width/threadsPerBlock.y);
        d_slowConvolution<<< numBlocks, threadsPerBlock>>>(d_img, d_result, d_kernel, width, height, radius);
        checkCudaErrors(cudaDeviceSynchronize());

    return 0;

}
