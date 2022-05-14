/* Matrix vector product for GPU
   Roanak Baviskar */

#include <iostream>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <ctime>

/* Use Matrix Class! */
#include "mat.h"
#include "submat.h"

// Thread block size
#define BLOCK_SIZE 64


// Forward declaration of the mat mul kernel
__global__ void MatVecKernel(Matrix, float[], float[]); 

/* Matrix vector multiplication
   :inputs: Matrix A, Float Array B
   :outputs: Float Array C
 */
void MatVec(const Matrix A, const float B[], float C[])
{
   float *d_B, *d_C;
   int Gpu = 1; 
   int toDev = 1;
   int N = A.width; 
   //Load A and B to device memory 
   //Allocate Matrix C
   Matrix d_A(A.width, A.height, A.stride, Gpu);
   d_A.load(A, toDev);


   //allocate our memory on GPU 
   hipMalloc(&d_B, N*sizeof(float));
   hipMalloc(&d_C, N*sizeof(float));

   //Memory Transfer! 
   hipMemcpy(d_B, B, N*sizeof(float), hipMemcpyHostToDevice);
   hipMemcpy(d_C, C, N*sizeof(float), hipMemcpyHostToDevice);

   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   dim3 dimGrid(A.width / dimBlock.x, A.height/ dimBlock.y);

   MatVecKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

   hipMemcpy(C, d_C, N*sizeof(float), hipMemcpyDeviceToHost);





   //Free device memory 
   d_A.dealloc(Gpu);
   hipFree(d_B);
	hipFree(d_C);

}

__global__ void MatVecKernel(Matrix A, float B[], float C[])
{
   // Each thread block calculates BLOCKSIZE elements of C
   __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
   __shared__ float Bs[BLOCK_SIZE]; // We will need a Blocksize chunk of B

   
   // Block row;
	int blockRow = blockIdx.y;

   // Each thread computes one element of Csub
	// By accumulating results into Cvalue
	float Cvalue = 0.0f;

   //Thread row and column index within the submatrix
	int row = threadIdx.y;
	int col = threadIdx.x;

   // Bs does not vary by M since M changes A's column and B only depends on row
   Bs[row] = B[blockRow * BLOCK_SIZE + row];

   // Loop over submatrices of A as big as the Blocks get that 
   // are needed for Cvalue. 
	for (int m = 0; m < (A.width/BLOCK_SIZE); m++) {

      //Get A submatrix
		subMatrix Asub(A, BLOCK_SIZE, blockRow, m);

      //Load Asub and Bsub from global memory into shared; 
      // Asub already cut by m, so no need to account for it
		As[row][col] = Asub.GetElem(row,col); 

      //Always sync threads when loading shared memory before doing computation
		__syncthreads();

		//Multiply the submatrices
		for (int e = 0; e < BLOCK_SIZE; e++)
			Cvalue += As[row][e]*Bs[row];

		//synchronize to make sure all threads are done computing
		__syncthreads();
   }
   // Fill value in C
	//each thread writes one element
   C[blockRow * BLOCK_SIZE + row] = Cvalue;  
}


void serialMatVec(const Matrix A, const float B[], float C[])
{
   for(int i = 0; i < A.width; i++)
   {
	float Cvalue = 0.0f;
	for(int j = 0; j < A.width; j++)
	{
		Cvalue += A.elements[i*A.width + j]*B[j];
	}
	C[i] = Cvalue;
   }
}


//Main program 
int main()
{
   // Set up matrices
   int Cpu = 0;
   int N = 1024;

   Matrix A(N, N, N, Cpu);
   float B[N];
   float C[N];
   std::fill_n (B, N, 1.f);
   for( int i = 0; i < A.height; i++){
      for( int j = 0; j < A.width; j++)
      {
            A.elements[i*A.stride + j] = 1.0f;
      }
   }

   MatVec(A, B, C);

   A.dealloc();


}
