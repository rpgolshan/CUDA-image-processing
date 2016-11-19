#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "convolution.cuh"

// 2D float texture
//texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;


/* must basic convolotion method in parallel
 * SLOW
 */
__global__ void d_slowConvolution(unsigned int *d_img, unsigned int *d_result, float *d_kernel, int width, int height, int radius)
{
//    float scale = 1.0f / (float)((r << 1) + 1);
//    int weightsum = 1;
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int loc = x + (blockIdx.y*blockDim.y)*width + threadIdx.y*width;
    uint3 accumulation = make_uint3(0,0,0);
    uint3 value = make_uint3(0, 0, 0);
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
               // value = 0;
                return;
            else { 
                unsigned int t = d_img[loc + i + j * width];
                //blue, green, red
                value = make_uint3(t & 0xff, (t>>8)&0xff, (t>>16)&0xff);
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
        if (accumulation.x > 0xff) accumulation.x = 0xff;
        if (accumulation.y > 0xff) accumulation.x = 0xff;
        if (accumulation.z > 0xff) accumulation.x = 0xff;

        unsigned int total = (accumulation.x & 0xff) + ((accumulation.y & 0xff) << 8) + ((accumulation.z & 0xff) << 16);
        d_result[loc] = total;
        printf("orig: %x new: %x ", d_img[loc], d_result[loc]);
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
