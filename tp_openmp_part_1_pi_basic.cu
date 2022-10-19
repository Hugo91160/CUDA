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

static long num_steps = 100000000;
double step;

__global__ void piAdd(double* x, double* sums, double step, int range){
	int i;

  for (i = (range * blockIdx.x) + 1; i <= (range * (blockIdx.x+1)); i++){
    *x = (i-0.5)*step;
    sums[blockIdx.x] = sums[blockIdx.x] + 4.0/(1.0+(*x)*(*x));
  }
}

int main (int argc, char** argv)
{
    
      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
    
    int blocks = 1000;
    int threads = 1;
	  double pi = 0.0;
	  
      step = 1.0/(double) num_steps;

	  //Allocate host memory
	  double *h_x = (double*)malloc(sizeof(double) * (num_steps/blocks));
    double *h_sum = (double*)malloc(sizeof(double) * (num_steps/blocks));
    *h_x = 0.0;
    *h_sum = 0.0;

	  //Allocate device memory
	  double * d_x;
    double * d_sum;
	  cudaMalloc((void **)&d_x,sizeof(double) * (num_steps/blocks));
	  cudaMalloc((void **)&d_sum,sizeof(double) * (num_steps/blocks));

	  cudaMemcpy(d_x, h_x, sizeof(double) * (num_steps/blocks), cudaMemcpyHostToDevice);
	  cudaMemcpy(d_sum, h_sum, sizeof(double) * (num_steps/blocks), cudaMemcpyHostToDevice);


      // Timer products.
      struct timeval begin, end;

      gettimeofday( &begin, NULL );

	  piAdd<<<blocks, threads>>>(d_x, d_sum, step, num_steps / blocks);
    cudaDeviceSynchronize();
    cudaMemcpy(h_sum, d_sum, sizeof(double) * (num_steps/blocks), cudaMemcpyDeviceToHost);
    
    double sum = 0.0;
    for (int i=0; i<(num_steps/blocks); i++){
      sum += h_sum[i];
    }

	  pi = step * sum;

      
      gettimeofday( &end, NULL );

      // Calculate time.
      double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
      printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,time);
}
