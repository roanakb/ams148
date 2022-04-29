#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <ctime>
#include <unistd.h>


void serial_saxpy(int n, float a, const float x[], float y[])
{
        for(std::size_t i = 0; i < n; ++i) {
        	y[i] += a*x[i];
		// std::cout<<"i is " << i<<std::endl; 
        }
}

void time_saxpy(int n)
{
    //create pointers and device
	float *d_x, *d_y; 
	
	const float a = 2.0f;

	//allocate and initializing memory on host
	std::vector<float> x(n, 1.f);
	std::vector<float> y1(n, 1.f);
	//float y2[N] = { 1.f };

	float y2[n];
	std::fill_n (y2, n, 1.f);

	clock_t sstart = clock();	// Serial Start
	serial_saxpy(n, a, x.data(), y2);
	// sleep(10);
	clock_t send = clock();		// Serial End
	float serial = float(send - sstart);
        std::cout<< " Serial Saxpy time for n = " << n << " is" << '\t' << serial << "ms" << std::endl; 
}

int main()
{

	int N[] = {16, 128, 1024, 2048, 65536}; 
	for(int i = 0; i < 5; ++i) {
        time_saxpy(N[i]);
    }

	std::cout<<"Done!"<<std::endl;  
	return 0;
}
