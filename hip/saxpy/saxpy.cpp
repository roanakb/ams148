#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <hip/hip_runtime.h>
#include <ctime>
#include <unistd.h>

/* this is the vector addition kernel. 
   :inputs: n -> Size of vector, integer
            a -> constant multiple, float
	    x -> input 'vector', constant pointer to float
	    y -> input and output 'vector', pointer to float  */
__global__ void saxpy(int n, float a, const float x[], float y[])
{
	int id = threadIdx.x + blockDim.x*blockIdx.x; /* Performing that for loop */ 
	// check to see if id is greater than size of array
	if(id < n){
		 y[id] +=  a*x[id]; // y[id] = y[id] + a*x[id]; 
	} 
}

void serial_saxpy(int n, float a, const float x[], float y[])
{
        for(std::size_t i = 0; i < n; ++i) {
        	y[i] += a*x[i];
		// std::cout<<"i is " << i<<std::endl; 
        }
}

int main()
{
	int N = 65536; 
	//create pointers and device
	float *d_x, *d_y; 
	
	const float a = 2.0f;

	//allocate and initializing memory on host
	std::vector<float> x(N, 1.f);
	std::vector<float> y1(N, 1.f);
	//float y2[N] = { 1.f };

	float y2[N];
	std::fill_n (y2, N, 1.f);

	// y2 = float[N]; //C++
/*
	float *x, *y; 
	x = new float[N]; //C++
	(*float)Malloc(x, N*sizeof(float)); //C
*/
	//allocate our memory on GPU 
	hipMalloc(&d_x, N*sizeof(float));
	hipMalloc(&d_y, N*sizeof(float));
	
	//Memory Transfer! 
	hipMemcpy(d_x, x.data(), N*sizeof(float), hipMemcpyHostToDevice);
	hipMemcpy(d_y, y1.data(), N*sizeof(float), hipMemcpyHostToDevice); 

	// Use HIP Events for timing

        hipEvent_t start, stop; 
        float time; 
        hipEventCreate(&start); 
        hipEventCreate(&stop); 
        hipEventRecord(start, 0); 


	//Launch the Kernel! In this configuration there is 1 block with 256 threads
	//Use gridDim = int((N-1)/256) in general  
	int gridDim = int(N/1024);
	saxpy<<<gridDim, 1024>>>(N, a, d_x, d_y);


        hipEventRecord(stop, 0); 
        hipEventSynchronize(stop); 
        hipEventElapsedTime(&time, start, stop); 
        std::cout<< " GPU Saxpy time =" << '\t'  << time << "ms" << std::endl; 

	//Transfering Memory back! 
	hipMemcpy(y1.data(), d_y, N*sizeof(float), hipMemcpyDeviceToHost);
	std::cout<<"First Element of z = ax + y is " << y1[0]<<std::endl; 
	hipFree(d_x);
	hipFree(d_y);

	clock_t sstart = clock();	// Serial Start
	serial_saxpy(N, a, x.data(), y2);
	sleep(10);
	clock_t send = clock();		// Serial End
	float serial = float(send - sstart);
        std::cout<< " Serial Saxpy time =" << '\t' << serial << "ms" << std::endl; 

	std::cout<<"Done!"<<std::endl;  
	return 0;
}
