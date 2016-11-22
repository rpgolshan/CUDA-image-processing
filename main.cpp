/*
 * Created by Rob Golshan
 * gl code and helper functions taken from NVIDIA sample code 
 * Demos common image filters using parallel gpu algorithms
 */

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

#include "kernels.h"
#include "convolution.cuh"

int weight = 1;
int radius = 1;

const char *image_filename = "./data/lena.ppm";
int filter = 0;
int type = 0;
unsigned int iterations = 1;


unsigned int width, height;
unsigned int *h_img = NULL;
unsigned int *d_img = NULL;
//unsigned int *d_result = NULL;
int *k = k0;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint texid = 0;   // texture

StopWatchInterface *timer = 0;

// display results using OpenGL
void display()
{

    // execute filter, writing results to pbo
    unsigned int *d_result;
    checkCudaErrors(cudaGLMapBufferObject((void **)&d_result, pbo));
    convolution(d_img, d_result, k, width, height, radius, type, weight, iterations); 
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
}

void idle()
{
    glutPostRedisplay();
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    checkCudaErrors(cudaFree(d_img));
    if (pbo)
    {
        checkCudaErrors(cudaGLUnregisterBufferObject(pbo));
        glDeleteBuffers(1, &pbo);
    }

    if (texid)
    {
        glDeleteTextures(1, &texid);
    }
}

const char *s = "identity";
int prev_type= 0;
void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
                glutDestroyWindow(glutGetWindow());
                return;
            break;
        case '=':
        case '+':
            if (type == 3)
                radius++;
            else if (filter == 9)
                radius >= 9? radius=9: radius++;
            break;
        case '-':
        case '_':
            if (filter == 9)
                radius <= 0? radius=0: radius--;
            break;
        case '[':
            iterations <= 1? iterations=1: iterations--;
            break;
        case ']':
            iterations+=1;
            break;
        case '0':
            filter = 0;
            weight = 1;
            radius = 1;
            k = k0;
            s = "identity";
            type = prev_type;
            break;
        case '1':
            filter = 1;
            weight = 13;
            radius = 2;
            k = k1;
            s = "blur";
            type = prev_type;
            break;
        case '2':
            filter = 2;
            weight = 9;
            radius = 4;
            k = k2;
            s = "motion blur";
            type = prev_type;
            break;
        case '3':
            filter = 3;
            weight = 1;
            radius = 2;
            k = k3;
            s = "detect horizontol edges";
            type = prev_type;
            break;
        case '4':
            filter = 4;
            weight = 1;
            radius = 2;
            k = k4;
            s = "detect vertical edges";
            type = prev_type;
            break;
        case '5':
            filter = 5;
            weight = 1;
            radius = 1;
            k = k5;
            s = "detect all edges";
            type = prev_type;
            break;
        case '6':
            filter = 6;
            weight = 1;
            radius = 1;
            k = k6;
            s = "sharpen";
            type = prev_type;
            break;
        case '7':
            filter = 7;
            weight = 273;
            radius = 2;
            k = guass;
            s = "guassian blur";
            type = prev_type;
            break;
        case '8':
            filter = 8;
            weight = 1;
            radius = 1;
            k = k8;
            s = "emboss";
            type = prev_type;
            break;
        case '9':
            filter = 9;
            radius = 9;
            weight = ((radius<<1)+1)*((radius<<1)+1);
            k = k9;
            s = "box filter (max radius 9)";
            type = prev_type;
            break;
        case 'q':
            prev_type = 0;
            type = 0;
            break;
        case 'w':
            prev_type = 1;
            type = 1;
            break;
        case 'e':
            filter = 9;
            type = 3;
            s = "fast box filter (independent of radius, no limit on radius)";
            break;
        case 'r':
            filter = 9;
            type = 2;
            k = k9;
            s = "separable box filter (max radius 9)";
            break;
        case 't':
            filter = 7;
            type = 2;
            radius = 2;
            k = guass;
            s = "separable guassian blur";
            break;
        default:
            break;
    }

    printf("filter: %s   convolution func: %d  radius: %d iterations:%d   ", s, type, radius, iterations);

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
    unsigned int ksize =  (2*radius+1)*(2*radius+1) * sizeof(int);

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &d_img, size));
    checkCudaErrors(cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice));
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
    glutIdleFunc(NULL); //IDLE here so its not just endlessly computing

    glutCloseFunc(cleanup);

    printf("Press '+' and '-' to change filter width\n");
    printf("0, 1, 2 - change filter order\n");
    printf("a = slow convolution, b = slow convolution w/ shared memory\n");

    if (!isGLVersionSupported(2,0) || !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }
}

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

    //PPM images only
    sdkLoadPPM4ub(image_path, (unsigned char **)&h_img, &width, &height);

    if (!h_img)
    {
        printf("Error unable to load PPM file: '%s'\n", image_path);
        exit(EXIT_FAILURE);
    }

    initGL(&argc, argv);
    findCudaGLDevice(argc, (const char **)argv);
    printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);
    
    initCudaBuffers();

    initGLBuffers();
    glutMainLoop();
    exit(EXIT_SUCCESS);
}

