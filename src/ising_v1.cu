#include "../inc/ising.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#define BLOCK_SIZE  32


//functions
__global__ void calculateNewSpinKernel(int * M, int * newM, double * w, int n, int * flag);


void ising( int *G, double *w, int k , int n)
{
  struct timeval startwtime, endwtime;
  double time = 0;
  //flag to terminate if no changes are made
  int terminate_flag;
  int * d_terminate_flag;
  //for pointer swap
  int * temp;

  //cuda
  int  * d_G, *d_newG;
  double * d_w;

  //cuda mallocs
  cudaMalloc(&d_terminate_flag,sizeof(int));
  cudaMalloc(&d_G, n*n*sizeof(int));
  cudaMalloc(&d_newG, n*n*sizeof(int));
  cudaMalloc(&d_w, 25*sizeof(double));

  //cuda memcpy G and W
  cudaMemcpy(d_G, G, n*n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, 25*sizeof(double), cudaMemcpyHostToDevice);

  //declare block size and grid size
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  int GRID_SIZE = ((n + BLOCK_SIZE -1)/BLOCK_SIZE);
  dim3 blocksPerGrid(GRID_SIZE, GRID_SIZE);



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
      //we will always memcpy here
      //numIter = i;
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

  //indeces
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  int col = blockIdx.y*blockDim.y+threadIdx.y;
  int thread_id = col* n + row;

  double influence = 0;

  int y = thread_id / n;
  int x = thread_id % n;

  //guard for extra threads
  if(thread_id < n*n)
  {

    for (int k=-2; k<=2;k++)
    {
      for(int l= -2; l<=2; l++)
      {
        influence += w[(2+k)*5+(2+l)] * M[((k + y + n) % n) * n + (l + x + n) % n];
      }
    }


  }

  //influence float point error
  if(fabs(influence) < 10e-7)
  {
    newM[thread_id] = M[thread_id];
  }
  else if(influence>0)
  {
    if(M[thread_id]!=1)
      *flag = 0;
      newM[thread_id] = 1;
  }
  else if(influence<0)
  {
    if(M[thread_id]!=-1)
      *flag = 0;
      newM[thread_id] = -1;
  }

}



