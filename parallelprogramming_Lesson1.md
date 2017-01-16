#Parallel Programming Notes

#Data Transfer
Data is transferred from CPU to GPU and from GPU to CPU using memcpy 

	cudaMemcpy
	
#Allocate GPU Memory
GPU memory is allocated using malloc function

	cudaMalloc
#Free GPU Memory	
Free GPU memory using free function

	cudaFree
#A CUDA Program structure
1. CPU allocates storage on GPU  - **cudaMalloc**
2. CPU copies INPUT data from CPU to GPU -  **cudaMemcpy**
3. CPU launches kernel on GPU to process the data - **Kernel Launch**
4. CPU copies result back to CPU from GPU - **cudaMemcpy**

#What's GPU good at?
1. Efficiently launch lot of threads
2. Running lot of threads in parallel

##Simple Example

Input   : Float Array  [0 ,1, 2, 3, ................. ,63]
Output array is square of input array.
Output: Float Array  [0^2  ,1^2 ,3 ^2 ,......... ,63^2]

###CPU code to square each element in an array
	for(i=0; i < 64; i++)
	{
		out[i] = in[i] * in[i];
	}
- Only one thread is involved
- No parallelism involved
- Total 64 Multiplication will be done
- If one multiplication takes 2ns then total will be 64*2ns = 128ns
###GPU code High Level View
It has two parts 
1.  Runs on GPU
2. Runs on CPU

###GPU
**Says nothing about degree of parallelism**

*Express OUT = IN * IN*

- Total 64 Multiplication will be done
- If one multiplication takes 10ns all 64 multiplication will run parallel so for 64 multiplication it will take only 10ns itself.

###CPU
- Allocate memory
- Copy data to/from GPU
- Launch Kernel -specifies degree of parallelism
	- cpu code : squarekernel<<<64>>> (outarray,inarray)
	- launches 64 threads
###CUDA Program to square

	#include <stdio.h>
	
	__global__ void square(float *d_out , float *d_in)
	{
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f ;
	}
	
	
	int main(int argc, char **argv)
	{
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	//generate the input array on the host/CPU
	float h_in[ARRAY_SIZE];
	float h_out[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
	 {
	        h_in[i] = float(i);
	 }
	
	
	//declare GPU memory pointers
	float * d_in;
	float * d_out;
	
	// allocate GPU memory
	cudaMalloc((void**) &amp;d_in, ARRAY_BYTES);
	cudaMalloc((void**) &amp;d_out, ARRAY_BYTES);
	
	//Transfer the array to GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES , cudaMemcpyHostToDevice);
	
	// Launch the Kernel
	square<<< 1, ARRAY_SIZE >>>(d_out, d_in);
	
	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	// Print the result from CPU
	        for(int i=0 ; i< ARRAY_SIZE ; i++)
	        {
	        printf("%f ", h_out[i]);
	        printf(((i % 4) != 3) ? "\t" : "\n");
	        }
	printf("\n");
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
	
	} //end of main		

CUDA Convention:
---------------------
Naming varibales like this avoids error and differentiates CPU and GPU variables.

- h_in - Data on th CPU the host 
- d_in - Data on the GPU the device
-----------
	//Transfer the array to GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES , cudaMemcpyHostToDevice);	
	
The fourth parameter cudaMemcpyHostToDevice defines to copy from host to device or vice versa.

	// Launch the Kernel
	square<<< ARRAY_SIZE >>>(d_out, d_in);

- CUDA Launch operator - CUDA Launch operator is indicated by **<<< >>> **signs
- It means launch the kernel named square on one block with 64 elements.
- The arguments to the kernel are two pointer d_in and d_out.
- This tell CPU to launch 64 kernels on GPU on 64 threads.
------
	__global__ void square(float *d_out , float *d_in)
	{
		int idx = threadIdx.x;
		float f = d_in[idx];
		d_out[idx] = f * f ;
	}
	
- ** ___global__ **  is the keyword by which the CUDA knows this is a kernel code.
- **int idx = threadIdx.x;**  - Each thread knows its ID. CUDA has an inbuilt variable called threadId

###CUDA Program to get cube of array

	#include <stdio.h>
	
	__global__ void cube(float *d_out , float *d_in)
	{
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f * f;
	}
	
	
	int main(int argc, char **argv)
	{
	const int ARRAY_SIZE = 96;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	
	//generate the input array on the host/CPU
	float h_in[ARRAY_SIZE];
	float h_out[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
	 {
	        h_in[i] = float(i);
	 }
	
	
	//declare GPU memory pointers
	float * d_in;
	float * d_out;
	
	// allocate GPU memory
	cudaMalloc((void**) &amp;d_in, ARRAY_BYTES);
	cudaMalloc((void**) &amp;d_out, ARRAY_BYTES);
	
	//Transfer the array to GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES , cudaMemcpyHostToDevice);
	
	// Launch the Kernel
	cube<<<1,ARRAY_SIZE >>>(d_out, d_in);
	
	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	// Print the result from CPU
	        for(int i=0 ; i< ARRAY_SIZE ; i++)
	        {
	        printf("%f ", h_out[i]);
	        printf(((i % 4) != 3) ? "\t" : "\n");
	        }
	printf("\n");
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
	
	} //end of main
	
Command to compile code
-------

	nvcc -o cube cube.cu

 - nvcc - Nvidia C Compiler.

##Configuring Kernel Launch

- 1 - means number of blocks.
- 64 - threads per block.

**square<<< 1, ARRAY_SIZE >>>(d_out, d_in);**

- It's capable of running blocks at a once.
- Each block has a maximum number of thread it can support.
	-  older GPU's  512 threads per block.
	- Newer GPU's 1024 threads per block.	
- So if 1280 threads is needed?
	- **square<<< 10, 128 >>>(d_out, d_in)**
	- **square<<< 5, 256 >>>(d_out, d_in)**
	- you cant't call ~~square<<< 1, 1280 >>>(d_out, d_in)~~ thats too many threads per block.
- CUDA supports2D and 3D blocks and threads which comes in image processing applications.
- **KERNEL<<< GRID OF BLOCKS, BLOCK OF Threads >>>(...)**
- 3 Dimesional
	- **square<<< dim3(bx,by,bz),dim3(tx,ty,tz),shmem  >>>(d_out, d_in)**
	- shmem - shared memory between blocks.
	- threadIdx - thread ID within block. eg: threadId.x,threadId.y
	- blockDim - size of block.
	- block idx - block within grid.
	- gridDim - size of grid.
	
Find no of blocks and threads
----------------------------------
**kernel<<< dim3(8,4,2),dim3(16,16)  >>>(...)**

- How many blocks? 
	- 8 * 4 * 2 = 64
- How many threads per block?
	- 16 * 16 = 256
- How many total threads?
	- 64 * 256 = 16384
	
###MAP
MAP is a key buliding block to GPU computing

- Set of elements to process [ 64 Floats ]
-  Function to run on each element [ "Square" ]
- Map's communication pattern
	- One element in One element out

**MAP( ELEMENTS, FUNCTION )**

Convert color image to balck & white
--------------------------
I = (R + G + B) / 3
Due to color sensitivity we can use the following equation
I = .299f * R + .587f * G + 	.114f  * B

CUDA progrmColor to Greyscale Conversion
--------------------

	// Homework 1
	// Color to Greyscale Conversion
	
	//A common way to represent color images is known as RGBA - the color
	//is specified by how much Red, Green, and Blue is in it.
	//The 'A' stands for Alpha and is used for transparency; it will be
	//ignored in this homework.
	
	//Each channel Red, Blue, Green, and Alpha is represented by one byte.
	//Since we are using one byte for each color there are 256 different
	//possible values for each color.  This means we use 4 bytes per pixel.
	
	//Greyscale images are represented by a single intensity value per pixel
	//which is one byte in size.
	
	//To convert an image from color to grayscale one simple method is to
	//set the intensity to the average of the RGB channels.  But we will
	//use a more sophisticated method that takes into account how the eye 
	//perceives color and weights the channels unequally.
	
	//The eye responds most strongly to green followed by red and then blue.
	//The NTSC (National Television System Committee) recommends the following
	//formula for color to greyscale conversion:
	
	//I = .299f * R + .587f * G + .114f * B
	
	//Notice the trailing f's on the numbers which indicate that they are 
	//single precision floating point constants and not double precision
	//constants.
	
	//You should fill in the kernel as well as set the block and grid sizes
	//so that the entire image is processed.
	
	#include "reference_calc.cpp"
	#include "utils.h"
	#include <stdio.h>
	
	__global__
	void rgba_to_greyscale(const uchar4* const rgbaImage,
	                       unsigned char* const greyImage,
	                       int numRows, int numCols)
	{
	  //TODO
	  //Fill in the kernel to convert from color to greyscale
	  //the mapping from components of a uchar4 to RGBA is:
	  // .x -> R ; .y -> G ; .z -> B ; .w -> A
	  //
	  //The output (greyImage) at each pixel should be the result of
	  //applying the formula: output = .299f * R + .587f * G + .114f * B;
	  //Note: We will be ignoring the alpha channel for this conversion
	
	  //First create a mapping from the 2D block and grid locations
	  //to an absolute 2D location in the image, then use that to
	  //calculate a 1D offset
	  
	    int threadsPerBlock = blockDim.x * blockDim.y;
	    int blockId = blockIdx.y + (blockIdx.x * gridDim.y);
	    int threadId = threadIdx.y + (threadIdx.x * blockDim.y);
	
	    int offset = (blockId * threadsPerBlock) + threadId;
	    greyImage[offset] = .299f*rgbaImage[offset].x + .587f*rgbaImage[offset].y + .114f*rgbaImage[offset].z;
	}
	
	void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
	                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
	{
	  //You must fill in the correct sizes for the blockSize and gridSize
	  //currently only one block with one thread is being launched
	  const dim3 blockSize(numRows/32+1, numCols/32+1);  //TODO
	  const dim3 gridSize( 32,32);  //TODO
	  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
	  
	  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}
	
	