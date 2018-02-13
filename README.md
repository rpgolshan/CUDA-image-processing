Created by Rob Golshan (uteid: rpg499)

Credit:
main.cpp: GL functions / idea of the image loop taken from NVIDIA cuda sample code
convolution.cu: Algorithms implemented based on convolutionSeparable.pdf in cuda sample code and a parallel implementation of algorithms in https://web.archive.org/web/20060718054020/http://www.acm.uiuc.edu/siggraph/workshops/wjarosz_convolution_2001.pdf

What does the program do?
========================
This is an implementation of several image processesing algorithms utilizing the parallelism of an NVIDIA GPU via CUDA. Algorithms implemented are:
1. 2D Convolution in parallel that works with any kernel (i.e. filter matrix)
  * O(radius^2) assuming all blocks run in parallel
2. 2D Convolution in parallel similar to #1, but uses shared memory. Works with any kernel.
  * This is faster than #1.
  * Shared memory requirements are (BLOCK_SIZE x kernel radius) squared
  * Could possibly be faster (but same time complexity) by loading the source image in a texture
3. 2D Convolution in parallel with SEPARABLE kernels ONLY.
  * Split into two functions that compute convolution of rows or convolution of columns
  * O(radius) assuming all blocks run in parallel
4. Boxfilter
  * Similar to #3, but uses properties of box filters to keep time low when using a big radius
  * O(width+height) assuming all blocks run in parallel
  * Could possibly be faster (but same time complexity) by loading the source image in a texture
  * Time taken independent of radius size
  * Multiple iterations of this similate a Guassian filter

Filters I purposely did not implement:
1. FFT filter
  * Requires more math knowledge than I currently have
  * Implementation would be padding kernel/image and using FFT library in cuda
  * Slower than separable implementation
  * Should only really be needed with using BIG kernels that are not separable
2. Guassian filters
  * We can either use a separable filter (#3) or a box filter several times (#4) to get the same result

Any other filters I didn't implement were either because I thought it was already a filter mentioned earlier, or I missed it in my research



Why use this over NVIDIA sample code? There is no reason. I doubt my implementations are any faster than the samples provided.

BUILDING
==========================
Build with make

Tested and built on a single GPU system with a GTX 980 (compute capability 5.2)
Have Xwindow system enabled to visually see results

Must either use the sample lena.ppm or have your own ppm image file

Running
========================
./convolution --image [path to image]

While the program is running and the XWindow is in focus, press h for a help command.
