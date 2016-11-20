#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "convolution.cuh"

// 2D float texture
//texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

#define BLOCK_SIZE 16 

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
                //value = make_uint3(0,0,0);
                continue;
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

/* The most basic convolution method in parallel
 * Takes advantage of shared memory in a GPU 
 * Can be used with any (square) kernel filter
 * FAST 
 * Each output pixel does radius^2 multiplications 
 * T = O(radius^2)
 * W = O(radius^2 * width * height)
 */
__global__ void d_sharedSlowConvolution(unsigned int *d_img, unsigned int *d_result, float *d_kernel, int width, int height, int radius)
{
    // Use a 1d array instead of 2D in order to coalesce memory access
    extern __shared__ unsigned int data[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // memory location in d_img
    const unsigned int loc = (blockIdx.x*blockDim.x + threadIdx.x) + (blockIdx.y*blockDim.y)*width + threadIdx.y*width;

    uint3 accumulation = make_uint3(0,0,0);
    uint3 value;
    float weight = 0.0f;


    int w = blockDim.x;
    int h = blockDim.y;

    /* to convolute the edges of a block, the shared memory must extend outwards of radius  */
#pragma unroll
    for (int i = -w; i <= w; i+= w) {
#pragma unroll
        for (int j = -h; j <= h; j+= h) {
            int x0 = threadIdx.x + i;
            int y0 = threadIdx.y + j;
            if (x0 < -radius || 
                x0 >= radius + w ||
                y0 < -radius ||
                y0 >= radius + h )
                continue;
            else {
                int newLoc = loc + i + j*width;
                if (newLoc < 0 || newLoc >= width*height)
                    data[threadIdx.x + i + radius + (threadIdx.y + j + radius)*(blockDim.x+radius*2)] = 0;
                else 
                    data[threadIdx.x + i + radius + (threadIdx.y + j + radius)*(blockDim.x+radius*2)] = d_img[loc + (i) + (j)*width];
            }
        }
        
    }

    __syncthreads();

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            unsigned int t = data[threadIdx.x + i + radius + (threadIdx.y + j + radius)*(blockDim.x+radius*2)];
            if (t == 0) continue;
            float temp = d_kernel[i + radius +  (j+radius)*(radius*2 + 1)];
            value = d_uintToRGB(t);
            value.x *= temp;
            value.y *= temp;
            value.z *= temp;
            weight += temp;
            accumulation += value;
        }
    }
    if (radius == 0) //i.e. original image
        d_result[loc] = data[threadIdx.x + radius +(threadIdx.y + radius)*(blockDim.x+radius*2)];
    else  {
        accumulation.x =  accumulation.x/weight;
        accumulation.y =  accumulation.y/weight;
        accumulation.z =  accumulation.z/weight;
        d_result[loc] = d_rgbToUint(accumulation);
    }
}



double convolution(unsigned int *d_img, unsigned int *d_result, float *d_kernel, int width, int height,
                 int radius, int type)
{
    checkCudaErrors(cudaDeviceSynchronize());

    // threadsPerBlock needs to be a multiple of 32 for proper coalesce
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(ceil((float)width / threadsPerBlock.x), ceil((float)height/threadsPerBlock.y));
    switch (type) {
        case 0: 
            d_slowConvolution<<< numBlocks, threadsPerBlock>>>(d_img, d_result, d_kernel, width, height, radius);
            break;
        case 1:
            d_sharedSlowConvolution<<< numBlocks, threadsPerBlock, (BLOCK_SIZE+radius*2)*(BLOCK_SIZE+radius*2)*sizeof(unsigned int)>>>(d_img, d_result, d_kernel, width, height, radius);
            break;
    }
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}
