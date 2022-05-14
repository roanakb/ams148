/*Matrix Transpose HIP
Roanak Baviskar */

#include <iostream>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <cstring>
#include <ctime>
#include <omp.h>

/* Use Matrix Class! */
#include "mat.h"
#include "submat.h"

// Thread block size
#define BLOCK_SIZE 64




// Forward declaration of the mat mul kernel
__global__ void TransposeKernel(const Matrix, Matrix); 
__global__ void naivekernel(const Matrix, Matrix); 


__global__ void naivekernel(const Matrix A, Matrix B)
{
	// Each Thread computes one element of B
	// by getting element from A
	float Cvalue = 0.0f;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x; 
	Cvalue += A.elements[row*A.width + col];
	B.elements[row*B.width + col] = Cvalue;
}

void NaiveTranspose(const Matrix A, Matrix B)
{

    int Gpu=1, toDev = 1, fromDev = 2; 
	//Load A to device memory
    Matrix d_A(A.width, A.height,0, Gpu);
    d_A.load(A, toDev); 

	// Allocate B in device memory
    Matrix d_B(B.width, B.height,0, Gpu);

    // Invoke kernel 
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);

    // Use hipEvent type for timing

    hipEvent_t start, stop; 
    float elapsed_secs; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0); 

    naivekernel<<<dimGrid, dimBlock>>>(d_A, d_B);
    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&elapsed_secs, start, stop); 
    std::cout<<" Naive GPU MatMul Time = "<< elapsed_secs << "ms" << std::endl;
    // Read B from device memory 
    B.load(d_B, fromDev); 
    // Free device memory 
    d_A.dealloc(Gpu);
    d_B.dealloc(Gpu);
}

//Main program 
int main()
{
// Set up matrices
    int Cpu = 0;
    int N = 1024;
    int M = 1024;

    Matrix A(N, M, N, Cpu), B(M, N, M, Cpu), C(N, N, N, Cpu);
    Matrix Ds(N, M, N, Cpu), D(N,M,N, Cpu);
    Matrix nC(N, N, N, Cpu); 
	

	//set values for A
    for( int i = 0; i < A.height; i++){
    	for( int j = 0; j < A.width; j++)
    	{
            A.elements[i*A.stride + j] = 1.0f;
    	}
    }


// Call matrix multiplication. 

//OpenMP
/*
	clock_t begin = clock();	
	CPUMatMul(A,B,D);
	clock_t end = clock();
	double fullcpu = double(end - begin) / (CLOCKS_PER_SEC*12);
	std::cout<< " CPU Time = " << fullcpu << "s" << std::endl; //*/
//Naive CUDA
	NaiveTranspose(A,B);

//SharedMemCUDA
	//MatMul(A,B,C);
	

//Deallocate Memory
    A.dealloc();
    B.dealloc();
    C.dealloc();
    Ds.dealloc();
    D.dealloc();
    nC.dealloc(); 
}
