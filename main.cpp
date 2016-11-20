/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


/*
 * This sample demonstrates how 2D convolutions
 * with very large kernel sizes
 * can be efficiently implemented
 * using FFT transformations.
 */


// Include CUDA runtime and CUFFT
#include <cufft.h>

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>



// CUDA includes and interop headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>      // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h>   // includes cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "convolution.cuh"

#define MAX(a,b) ((a > b) ? a : b)

#define USE_SIMPLE_FILTER 0

#define MAX_EPSILON_ERROR 5.0f
#define THRESHOLD  0.15f




float box[] =
{
  1, 1, 1,
  1, 2, 1,
  1, 1, 1
};




/*
float box[] =
{
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
*/


/*
float box[] =
{
  0, 0, 1,
  0, 0, 0,
  1, 0, 0
};
*/


/*
 * identity matrix
 */

/*
float box[] =
{
  0, 0, 0,
  0, 1, 0,
  0, 0, 0
};

*/

int weight = 9;
int radius = 1;
/*
 * START OF NVIDIA CODE//
 */                    
// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "lena_10.ppm",
    "lena_14.ppm",
    "lena_18.ppm",
    "lena_22.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_10.ppm",
    "ref_14.ppm",
    "ref_18.ppm",
    "ref_22.ppm",
    NULL
};

const char *image_filename = "./data/part1pairs/sign_1.ppm";
int filter = 1;
int type = 0;
int nthreads = 64;  // number of threads per block

unsigned int width, height;
unsigned int *h_img = NULL;
unsigned int *d_img = NULL;
//unsigned int *d_result = NULL;
float *d_kernel = NULL;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint texid = 0;   // texture

StopWatchInterface *timer = 0;

// display results using OpenGL
void display()
{

    // execute filter, writing results to pbo
    unsigned int *d_result;
    checkCudaErrors(cudaGLMapBufferObject((void **)&d_result, pbo));
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    switch (filter) {
        case 1: 
            convolution(d_img, d_result, d_kernel, width, height, radius, type); 
            break;
        case 0:
            convolution(d_img, d_result, d_kernel, width, height, 0, type); 
            break;
    }
    sdkStopTimer(&timer);
    printf("time taken: %f\n", sdkGetTimerValue(&timer));
//    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGLUnmapBufferObject(pbo));

    // load texture from pbo
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, texid);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 1);
    glVertex2f(0, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 1);
    glTexCoord2f(0, 0);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glutSwapBuffers();


    //computeFPS();
}

void idle()
{
    glutPostRedisplay();
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    checkCudaErrors(cudaFree(d_img));
    checkCudaErrors(cudaFree(d_kernel));
//    checkCudaErrors(cudaFree(d_result));

    //if (!runBenchmark)
    //{
        if (pbo)
        {
            checkCudaErrors(cudaGLUnregisterBufferObject(pbo));
            glDeleteBuffers(1, &pbo);
        }

        if (texid)
        {
            glDeleteTextures(1, &texid);
        }
    //}
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
                glutDestroyWindow(glutGetWindow());
                return;
            break;

        case '=':
            radius+=1;
            break;

        case '-':
            radius-=2;

            if (radius < 0)
            {
                radius = 0;
            }

            break;

        case '+':
            radius+=1;
            break;

        case '_':
            radius-=1;

            if (radius < 0)
            {
                radius = 0;
            }

            break;

        case '0':
            filter = 0;
            break;

        case '1':
            filter = 1;
            break;
        case 'a':
            type = 0;
            break;
        case 'b':
            type = 1;
            break;

        default:
            break;
    }

    printf("convolution func = %d           radius = %d\n", type, radius);

    glutPostRedisplay();
}



void reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}



void initCudaBuffers()
{
    unsigned int size = width * height * sizeof(unsigned int);
    unsigned int ksize =  (2*radius+1)*(2*radius+1) * sizeof(float);

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &d_img, size));
    checkCudaErrors(cudaMalloc((void **) &d_kernel, ksize));
//    checkCudaErrors(cudaMalloc((void **) &d_result, size));

    checkCudaErrors(cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel, box, ksize, cudaMemcpyHostToDevice));

    sdkCreateTimer(&timer);
}



void initGLBuffers()
{
    // create pixel buffer object to store final image
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, h_img, GL_STREAM_DRAW_ARB);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    checkCudaErrors(cudaGLRegisterBufferObject(pbo));

    // create texture for display
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA image processing");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(NULL);
    //glutIdleFunc(idle);

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

    printf("Press '+' and '-' to change filter width\n");
    printf("0, 1, 2 - change filter order\n");
    printf("a = slow convolution, b = slow convolution w/ shared memory\n");

    if (!isGLVersionSupported(2,0) || !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }
}
/* 
 * END OF NVIDIA CODE
 */



int main(int argc, char **argv)
{
    setenv ("DISPLAY", ":0", 0);
    printf("[%s] - Starting...\n", argv[0]);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char **)argv);

    int nFailures = 0;

    // Get the path of the filename
    char *filename;

    if (getCmdLineArgumentString(argc, (const char **) argv, "image", &filename))
    {
        image_filename = filename;
    }

    // load image
    char *image_path = sdkFindFilePath(image_filename, argv[0]);

    if (image_path == NULL)
    {
        fprintf(stderr, "Error unable to find and load image file: '%s'\n", image_filename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPPM4ub(image_path, (unsigned char **)&h_img, &width, &height);

    if (!h_img)
    {
        printf("Error unable to load PPM file: '%s'\n", image_path);
        exit(EXIT_FAILURE);
    }

    initGL(&argc, argv);
    findCudaGLDevice(argc, (const char **)argv);
    printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);
    
    /*
    for (int i = 0; i < width*height; i++) {
        printf("%d, ", h_img[i]);
    }
    printf("\n ");
    */
    initCudaBuffers();
//    convolution(d_img, d_result, d_kernel, width, height, radius); 

    initGLBuffers();
    glutMainLoop();
    exit(EXIT_SUCCESS);
}

