/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <iostream>
#include <fstream>

static long num_steps = 100000000;
static int num_blocks = 1;
static int num_threads = 1;
double step;

__global__ void piAdd(float* sums, long num_steps){
      
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      double x, sum = 0.0;
      int stride = blockDim.x * gridDim.x;
      double step = 1.0/(double) num_steps;

      for (int i = tid; i < num_steps; i += stride){
        x = (i-0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
      }

      atomicAdd(sums, sum);

}


int main (int argc, char** argv)
{
    
    // Read command line arguments.
	for (int i = 0; i < argc; i++)
	{
		if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-num_steps") == 0))
		{
			num_steps = atol(argv[++i]);
			printf("  User num_steps is %ld\n", num_steps);
		}
		else if ((strcmp(argv[i], "-B") == 0) || (strcmp(argv[i], "-num_blocks") == 0))
		{
			num_blocks = atol(argv[++i]);
			printf("  User num_blocks is %ld\n", num_blocks);
		}
		else if ((strcmp(argv[i], "-T") == 0) || (strcmp(argv[i], "-num_threads") == 0))
		{
			num_threads = atol(argv[++i]);
			printf("  User num_threads is %ld\n", num_threads);
		}
		else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0))
		{
			printf("  Pi Options:\n");
			printf("  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n");
			printf("  -num_blocks (-B) <int>:      Number of blocks used to compute Pi (by default 1)\n");
			printf("  -num_threads (-T) <int>:      Number of threads per block to compute Pi (by default 1)\n");
			printf("  -help (-h):            print this message\n\n");
			exit(1);
		}
	}

	double pi = 0.0;
	  
    step = 1.0/(double) num_steps;

    // Allocate host memory
	float *h_sum = (float *)malloc(sizeof(float) * num_blocks);
	for (int i = 0; i < num_blocks; i++)
	{
		h_sum[i] = 0.0;
	}

	// Allocate device memory
	float *d_sum;
	cudaMalloc((void **)&d_sum, sizeof(float) * num_blocks);

	cudaMemcpy(d_sum, h_sum, sizeof(float) * num_blocks, cudaMemcpyHostToDevice);

    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );

    piAdd<<<num_blocks, num_threads>>>(d_sum, num_steps);
	cudaDeviceSynchronize();
	cudaMemcpy(h_sum, d_sum, sizeof(float) * num_blocks, cudaMemcpyDeviceToHost);

    pi = *h_sum * step;


    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
            1.0e-6 * ( end.tv_usec - begin.tv_usec );
            
    printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,time);

    std::fstream output;
	output.open("pi_stats.csv", std::ios_base::app);
	output << "very basic"
		   << ", " << num_blocks << ", " << num_threads << ", " << num_steps << ", " << time << "\n";
}
