#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <ctime>
#include <unistd.h>

// Thread block size
#define BLOCK_SIZE 64


void serialMatVec(int N, std::vector<float> A, const float B[], float C[])
{
   for(int i = 0; i < N; i++)
   {
	float Cvalue = 0.0f;
	for(int j = 0; j < N; j++)
	{
        // std::cout<<i*N + j<<std::endl; 
		Cvalue += A.at(i*N + j)*B[j];
	}
	C[i] = Cvalue;
   }
}



//Main program 
void time_serial_matvec(int N)
{
   // Set up matrices
   int Cpu = 0;
   int s = 0;

	std::vector<float> A(N*N, 1.f);
   float B[N];
   float C[N];
   std::fill_n (B, N, 1.f);

   /*
   for( int i = 0; i < N; i++){
      for( int j = 0; j < N; j++)
      {
            A[i*N + j] = 1.0f;
      }
   }
   */

//Serial 
	clock_t sstart = clock();	//Serial Start
	serialMatVec(N,A,B,C);
	clock_t send = clock(); 	//Serial End
	double serial = double(send - sstart) / CLOCKS_PER_SEC;	
    std::cout<< " Serial Matrix-Vector Product time for n = " << N << " is" << '\t' << serial << "ms" << std::endl; 
}

int main()
{
	int N[] = {16, 128, 1024, 2048, 65536}; 
	for(int i = 0; i < 5; ++i) {
        time_serial_matvec(N[i]);
    }
	std::cout<<"Done!"<<std::endl;  
	return 0;
}
