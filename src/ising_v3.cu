#include "../inc/ising.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda.h"
#include <sys/time.h>
#define TILE_SIZE 16
#define BLOCK_SIZE  (TILE_SIZE + 4)


//functions
__global__ void calculateNewSpinKernel(int * M, int * newM, double * w, int n, int * flag);
__constant__ double shared_w[25];

void ising( int *G, double *w, int k , int n)
{
  struct timeval startwtime, endwtime;
  double time = 0;
  //flag to terminate if no changes are made
  int   terminate_flag;
  int * d_terminate_flag;
  //for pointer swap
  int * temp;

  //cuda
  int  * d_G, *d_newG;
  double * d_w;

  //load shared weights to memory
  double temp_w[25];
  for(int k = 0;k<25;k++)
  {
    temp_w[k] = w[k];
  }
  cudaMemcpyToSymbol(shared_w, temp_w, 25*sizeof(double));

  //cuda mallocs
  cudaMalloc(&d_terminate_flag,sizeof(int));
  cudaMalloc(&d_G, n*n*sizeof(int));
  cudaMalloc(&d_newG, n*n*sizeof(int));
  cudaMalloc(&d_w, 5*5*sizeof(double));

  //cuda memcpy G and W
  cudaMemcpy(d_G, G, n*n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, 25*sizeof(double), cudaMemcpyHostToDevice);

  //declare block size and grid size
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  // dim3 blocksPerGrid((n-1)/TILE_SIZE+1, (n-1)/TILE_SIZE+1);
  dim3 blocksPerGrid(17, 17);


  //k steps iterations
  for(int i= 0 ; i < k ;i++)
  {
    //reset flag value
    terminate_flag = 1;
    cudaMemcpy(d_terminate_flag, &terminate_flag,sizeof(int), cudaMemcpyHostToDevice);
    //call kernel
    gettimeofday (&startwtime, NULL);
    calculateNewSpinKernel<<<blocksPerGrid, threadsPerBlock>>>(d_G,d_newG,d_w,n,d_terminate_flag);
    cudaDeviceSynchronize();
    gettimeofday (&endwtime, NULL);
    time += (double)((endwtime.tv_usec - startwtime.tv_usec)
        /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
  
    //swap pointers
    temp = d_G;
    d_G = d_newG;
    d_newG = temp;  

    //we need device value for flag here
    cudaMemcpy(&terminate_flag, d_terminate_flag, sizeof(int), cudaMemcpyDeviceToHost);

    if (terminate_flag == 1)
    {
      break;

    }
    printf("Kernel time: %f seconds\n", time );
    cudaMemcpy(G,d_G, n*n*sizeof(int), cudaMemcpyDeviceToHost);


    //cudaFree

  }

    cudaFree(d_newG);
    cudaFree(d_G);
    cudaFree(d_w);


}


//kernel function
__global__ void calculateNewSpinKernel(int * M, int * newM, double * w, int n, int * flag)
{

  __shared__ int shared_G[(BLOCK_SIZE)][(BLOCK_SIZE)];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int y_out = blockIdx.y*TILE_SIZE + ty;
  int x_out = blockIdx.x*TILE_SIZE + tx;

  int y_in = y_out - 2;
  int x_in = x_out - 2; 



  if ((x_in >= 0) && (y_in >= 0) && (x_in < n) && (y_in < n))
        shared_G[ty][tx] = M[y_in * n + x_in];
    else 
        shared_G[ty][tx] = M[((y_in + n)%n)*n + (x_in + n)%n];

  __syncthreads();

  //only tilesize * tilesize threads calculate influence
  double influence = 0;
  if (ty < TILE_SIZE && tx < TILE_SIZE)
  {
    for(int i = 0; i < 5; i++) 
    {
      for(int j = 0; j < 5; j++)
      {
        influence += shared_w[i*5+j] * shared_G[ty+i][tx+j];
      }
    }
    //influence float point error
    if(fabs(influence) < 10e-7)
    {
      newM[y_out * n + x_out ] = shared_G[ty+2][tx+2];
    }
    else if(influence>0)
    {
      if(shared_G[ty+2][tx+2]!=1)
        *flag = 0;
      newM[y_out * n + x_out ] = 1;
    }
    else if(influence<0)
    {
    if(M[y_out * n + x_out ]!=-1)
      *flag = 0;
      newM[y_out * n + x_out] = -1;
    }
  }

  // //threads< n calculate output
  // if(y_out < n && x_out < n)
  // {
  //     //influence float point error
  //     if(fabs(influence) < 10e-7)
  //     {
  //       newM[y_out * n + x_out ] = M[y_out * n + x_out ];
  //     }
  //     else if(influence>0)
  //     {
  //     if(M[y_out * n + x_out ]!=1)
  //       *flag = 0;
  //       newM[y_out * n + x_out ] = 1;
  //     }
  //     else if(influence<0)
  //     {
  //     if(M[y_out * n + x_out ]!=-1)
  //       *flag = 0;
  //       newM[y_out * n + x_out] = -1;
  //     }

}