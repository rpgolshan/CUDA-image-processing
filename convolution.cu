#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "convolution.cuh"

// Kernel cannot have radius bigger than 15
__constant__ int d_kernel[1024];

#define BLOCK_SIZE 16 

/* 
 * Converts a uint to a uint3, seperating RGB
 * colors by every byte.
 * Most significat to least significant:
 * Red, Green, Blue
 */
__device__ __forceinline__ int3 d_uintToRGB(unsigned int orig)
{
    int3 rgb;
    rgb.x = orig & 0xff;
    rgb.y = (orig>>8)&0xff;
    rgb.z = (orig>>16)&0xff;
    return rgb;
}

/*
 * Converts a uint3 to an unsigned int
 * Assumes each vector member correspond to RGB colors
 * Truncates rgb colors bigger than 1 byte
 */
__device__ __forceinline__ unsigned int d_rgbToUint(int3 rgb)
{
    if (rgb.x > 0xff) rgb.x = 0xff;
    else if (rgb.x < 0) rgb.x = 0;
    if (rgb.y > 0xff) rgb.y = 0xff;
    else if (rgb.y < 0) rgb.y = 0;
    if (rgb.z > 0xff) rgb.z = 0xff;
    else if (rgb.z < 0) rgb.z = 0;

    return (rgb.x & 0xff) | ((rgb.y & 0xff) << 8) | ((rgb.z & 0xff) << 16);
}

__device__ __forceinline__ int3 d_divide(int3 orig, int op)
{
    orig.x = orig.x/op;
    orig.y = orig.y/op;
    orig.z = orig.z/op;
    return orig;
}

/* The most basic convolution method in parallel
 * Does not take advantage of memory optimizations with a GPU
 * Can be used with any (square) kernel filter
 * SLOW
 * Each output pixel does radius^2 multiplications 
 * T = O(radius^2)
 * W = O(radius^2 * width * height)
 */
__global__ void d_slowConvolution(unsigned int *d_img, unsigned int *d_result, int width, int height, int radius, int weight)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    const unsigned int loc =  x + y*width;
    int3 accumulation = make_int3(0,0,0);
    int3 value;

    if (x >= width || y >= height) return;
    assert(x < width);
    assert(y < height);
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            if ((x + i < 0) || //left side out of bounds
                (x + i >= width) || //right side OoB
                (y + j < 0) || //top OoB
                (y + j >= height)) //bot OoB
                continue;
            value = d_uintToRGB(d_img[loc + i + j * width]);
            int temp = d_kernel[i + radius +  (j+radius)*((radius << 1) + 1)];
            value *= temp;
            accumulation += value;
        }
    }
    accumulation = d_divide(accumulation, weight);
    d_result[loc] = d_rgbToUint(accumulation);
}

/* The most basic convolution method in parallel
 * Takes advantage of shared memory in a GPU 
 * Can be used with any (square) kernel filter
 * FAST 
 * Each output pixel does radius^2 multiplications 
 * T = O(radius^2)
 * W = O(radius^2 * width * height)
 */
__global__ void d_sharedSlowConvolution(unsigned int *d_img, unsigned int *d_result, int width, int height, int radius, int weight)
{
    // Use a 1d array instead of 2D in order to coalesce memory access
    extern __shared__ unsigned int data[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // memory location in d_img
    const unsigned int loc =  x + y*width;

    int3 accumulation = make_int3(0,0,0);
    int3 value;

    int w = blockDim.x;
    int h = blockDim.y;

    /* to convolute the edges of a block, the shared memory must extend outwards of radius  */
#pragma unroll 3 
    for (int i = -w; i <= w; i+= w) {
#pragma unroll 3
        for (int j = -h; j <= h; j+= h) {
            int x0 = threadIdx.x + i;
            int y0 = threadIdx.y + j;
            int newLoc = loc + i + j*width;
            if (x0 < -radius || 
                x0 >= radius + w ||
                y0 < -radius ||
                y0 >= radius + h || 
                newLoc < 0 ||
                newLoc >= width*height)
                continue;
            data[threadIdx.x + i + radius + (threadIdx.y + j + radius)*(blockDim.x+(radius << 1))] = d_img[newLoc];
        }
    }

    __syncthreads();

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            unsigned int t = data[threadIdx.x + i + radius + (threadIdx.y + j + radius)*(blockDim.x+(radius << 1))];
            int temp = d_kernel[i + radius +  (j+radius)*((radius << 1) + 1)];
            value = d_uintToRGB(t);
            value *= temp; 
            accumulation += value;
        }
    }
    accumulation = d_divide(accumulation, weight);
    d_result[loc] = d_rgbToUint(accumulation);
}

/* VERY FAST convolution method in parallel 
 * Takes advantage of shared memory in a GPU 
 * Can be used with ONLY WITH SEPERABLE kernel filters
 * Each output pixel does radius^2 multiplications 
 * T = O(radius + radius)
 * W = O(radius * width * radius*height)
 */
__global__ void d_sepRowConvolution(unsigned int *d_img, unsigned int *d_result, int width, int height, int radius)
{
    // Use a 1d array instead of 2D in order to coalesce memory access
    extern __shared__ unsigned int data[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // memory location in d_img
    const unsigned int loc = (blockIdx.x*blockDim.x + threadIdx.x) + (blockIdx.y*blockDim.y)*width + threadIdx.y*width;

    int3 accumulation = make_int3(0,0,0);
    int3 value;
    int weight = 0;


    int w = blockDim.x;

    /* to convolute the edges of a block, the shared memory must extend outwards of radius  */
#pragma unroll 3
    for (int i = -w; i <= w; i+= w) {
        int x0 = threadIdx.x + i;
        int newLoc = loc + i;
        if (x0 < -radius || 
            x0 >= radius + w ||
            newLoc < 0 || 
            newLoc >= width*height)
            continue;
        data[threadIdx.x + i + radius + (threadIdx.y) *(blockDim.x+(radius << 1))] = d_img[newLoc];
    }

    __syncthreads();

    for (int i = -radius; i <= radius; i++) {
        unsigned int t = data[threadIdx.x + i + radius + (threadIdx.y)*(blockDim.x+(radius << 1))];
        int temp = d_kernel[i + radius];
        value = d_uintToRGB(t);
        value *= temp;
        weight += temp;
        accumulation += value;
    }
    accumulation = d_divide(accumulation, weight);
    d_result[loc] = d_rgbToUint(accumulation);
}

/* VERY FAST convolution method in parallel 
 * Takes advantage of shared memory in a GPU 
 * Can be used with ONLY WITH SEPERABLE kernel filters
 * Each output pixel does radius^2 multiplications 
 * T = O(radius + radius)
 * W = O(radius * width * radius*height)
 */
__global__ void d_sepColConvolution(unsigned int *d_result, int width, int height, int radius)
{
    // Use a 1d array instead of 2D in order to coalesce memory access
    extern __shared__ unsigned int data[];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // memory location in d_img
    const unsigned int loc = (blockIdx.x*blockDim.x + threadIdx.x) + (blockIdx.y*blockDim.y)*width + threadIdx.y*width;

    int3 accumulation = make_int3(0,0,0);
    int3 value;
    float weight = 0.0f;


    int h = blockDim.y;

    /* to convolute the edges of a block, the shared memory must extend outwards of radius  */
#pragma unroll 3
    for (int j = -h; j <= h; j+= h) {
        int y0 = threadIdx.y + j;
        int newLoc = loc + j*width;
        if (y0 < -radius || 
            y0 >= radius + h ||
            newLoc < 0 ||
            newLoc >= width*height)
            continue;
            data[threadIdx.x + (threadIdx.y + j + radius)*(blockDim.x)] = d_result[newLoc];
    }

    __syncthreads();

    for (int j = -radius; j <= radius; j++) {
        unsigned int t = data[threadIdx.x + (threadIdx.y + j + radius)*(blockDim.x)];
        float temp = d_kernel[(j + radius)*((radius << 1)+1)];
        value = d_uintToRGB(t);
        value *= temp;
        weight += temp;
        accumulation += value;
    }
    accumulation = d_divide(accumulation, weight);
    d_result[loc] = d_rgbToUint(accumulation);
}


double convolution(unsigned int *d_img, unsigned int *d_result, int *h_kernel, int width, int height,
                 int radius, int type, int weight)
{
    checkCudaErrors(cudaDeviceSynchronize());

    // threadsPerBlock needs to be a multiple of 32 for proper coalesce
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(ceil((float)width / threadsPerBlock.x), ceil((float)height/threadsPerBlock.y));

    //copy kernel to device memory
    checkCudaErrors(cudaMemcpyToSymbol(d_kernel, h_kernel, ((radius << 1)+1)*((radius << 1)+1)*sizeof(int)));

    switch (type) {
        case 0: 
            d_slowConvolution<<< numBlocks, threadsPerBlock>>>(d_img, d_result, width, height, radius, weight);
            break;
        case 1:
            d_sharedSlowConvolution<<< numBlocks, threadsPerBlock, (BLOCK_SIZE+(radius << 1))*(BLOCK_SIZE+(radius << 1))*sizeof(unsigned int)>>>(d_img, d_result, width, height, radius, weight);
            break;
        case 2:
            d_sepRowConvolution<<< numBlocks, threadsPerBlock, (BLOCK_SIZE+(radius << 1))*(BLOCK_SIZE)*sizeof(unsigned int)>>>(d_img, d_result, width, height, radius);
            d_sepColConvolution<<< numBlocks, threadsPerBlock, (BLOCK_SIZE)*(BLOCK_SIZE+(radius << 1))*sizeof(unsigned int)>>>(d_result, width, height, radius);
            break;
    }
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}
