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


/* Shared Matrix Multiplication Routines */ 

/* Transpose with shared memory 
   :inputs: Matrix A
   :outputs: Matrix B = A^T
 */ 
void Transpose(const Matrix A, Matrix B)
{
    int Gpu = 1; 
    int toDev = 1, fromDev = 2; 
    //Load A and B to device memory 
    //Allocate Matrix C
    Matrix d_A(A.width, A.height, A.stride, Gpu);
    Matrix d_B(B.width, B.height, B.stride, Gpu);
    d_A.load(A, toDev);
	
    // Invoke Kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(A.width / dimBlock.x, A.height/ dimBlock.y); 
    //Use HIP Events for timing
    hipEvent_t start, stop; 
    float time; 
    hipEventCreate(&start); 
    hipEventCreate(&stop); 
    hipEventRecord(start, 0); 

    TransposeKernel<<<dimGrid, dimBlock>>>(d_A, d_B);
    hipEventRecord(stop, 0); 
    hipEventSynchronize(stop); 
    hipEventElapsedTime(&time, start, stop); 
    std::cout<< " Shared Memory Matrix-Vector Multiplication time for N = " 
             << A.width << " is " << '\t' 
             << time << "ms" << std::endl; 

	// Read B from Device memory 
    B.load(d_B, fromDev);
	
    //Free device memory 
    d_A.dealloc(Gpu);
    d_B.dealloc(Gpu);
}



// Matrix TransposeKernel
__global__ void TransposeKernel(Matrix A, Matrix B)
{
	//Static shared memory for Asub and Bsub
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];


	// Block row and column;
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

   int brow = 0;
   int bcol = 0;

   // IF BLOCK ROW == BLOCK COL, we can get same sub matrix of C
   // IF NOT, we need to place items in transposed location
   if (blockRow == blockCol) {
      brow = blockRow;
      bcol = blockCol;
   }
   else {
      brow = blockCol;
      bcol = blockRow;
   }
	//Thread block computes one sub matrix Csub of C
	subMatrix Bsub(B, BLOCK_SIZE,  brow, bcol);

	// Each thread computes one element of Csub
	// By accumulating results into Cvalue
	float Cvalue = 0.0f; 

	//Thread row and column index within the submatrix
	int row = threadIdx.y;
	int col = threadIdx.x; 

	// Loop over submatrices of A and B that are required for Csub
	//Multiply each pair of sub-matrices together
	//and summ the results
	for (int m = 0; m < (A.width/BLOCK_SIZE); m++){
		
		//Get A submatrix
		subMatrix Asub(A, BLOCK_SIZE, blockRow, m);

		//Load Asub and Bsub from global memory into shared; 

		As[row][col] = Asub.GetElem(row,col);

		//Always sync threads when loading shared memory before doing computation
		__syncthreads();

		//Set Cvalue to correct value
      Cvalue = As[row][col];

		//synchronize to make sure all threads are done computing
		__syncthreads();
	}
	//write Csub back into global memory 
	//each thread writes one element
	Bsub.SetElem(col, row, Cvalue);
}


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
    std::cout<< " Naive Matrix Transpose Multiplication time for N = " 
             << A.width << " is " << '\t' 
             << time << "ms" << std::endl;     // Read B from device memory 
    B.load(d_B, fromDev); 
    // Free device memory 
    d_A.dealloc(Gpu);
    d_B.dealloc(Gpu);
}

//Main program 
void time_transpose(int N)
{
// Set up matrices
    int Cpu = 0;
    int M = N;

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
	//NaiveTranspose(A,B);

//SharedMemCUDA
	Transpose(A,B);
	

//Deallocate Memory
    A.dealloc();
    B.dealloc();
    C.dealloc();
    Ds.dealloc();
    D.dealloc();
    nC.dealloc(); 
}

int main()
{
   int N[] = {16, 128, 1024, 2048, 65536}; 
	for(int i = 0; i < 5; ++i) {
        time_transpose(N[i]);
    }
	std::cout<<"Done!"<<std::endl;  
	return 0;
}
