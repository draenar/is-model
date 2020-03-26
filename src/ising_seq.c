#include "../inc/ising.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

//functions
void calculateNewSpin(int * M, int * newM, double * w, int n, int row,
	int col, int * flag);



void ising( int *G, double *w, int k , int n)
{
	struct timeval startwtime, endwtime;
  	double time = 0;
	//matrix to store new G
	int * newG = (int *)malloc(n*n*sizeof(int));
	if (newG == NULL)
	{
		printf("Memory allocation failed");
		return;
	}

	//flag to terminate if no changes are made
	int terminate_flag;
	//for pointer swap
	int * temp;

	int  numIter;

	//k steps iterations
	for(int i= 0 ; i < k ;i++)
	{
		//reset flag value
		terminate_flag = 1;
		gettimeofday (&startwtime, NULL);
		//calculate new G
		//for loop to calculate new spin for every point
		for(int j=0; j<n; j++)
		{
    		for(int k=0;k<n;k++)
    		{
      			calculateNewSpin(G,newG,w,n,j,k,&terminate_flag);
      		}
      	}
      	gettimeofday (&endwtime, NULL);
    	time += (double)((endwtime.tv_usec - startwtime.tv_usec)
        /1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
      	printf("Kernel time: %f seconds\n", time );
      	//swap pointers
      	temp = G;
    	G = newG;
    	newG = temp;	
 		

      	if (terminate_flag == 1)
      	{
      		break;
      		//keep number of iteration if it breaks earlien than k
      		numIter = i;
      	}

      	//fix pointer swap issue for odd number of iterations
      	if(((numIter+1)%2)!=0)
      	{
      		memcpy (newG, G,n*n*sizeof(int));
      	}


	}




}


void calculateNewSpin(int * M, int * newM, double * w, int n, int row,
	int col, int * flag)
{
	double influence = 0;
	//row indexes for modulo operations
	int row_index, col_index;



	for(int i=row-2;i<=row+2;i++ )
	{
		for(int j=col-2;j<=col+2;j++ )
		{
			//guard to leave middle point out
			if((i==row) && (j==col))
				continue;

			//modulo operations
			row_index = (i+n) % n;
			col_index = (j+n) % n;

			influence += M[row_index * n + col_index] * w[(2+i-row)* 5 + (2+j- col)];

		}
	}

	//influence float point error
	if(fabs(influence) < 10e-7)
	{
		newM[row * n + col] = M[row * n + col];
	}
	else if(influence>0)
	{
		if(M[row * n + col]!=1)
			*flag = 0;
		newM[row*n + col] = 1;
	}
	else if(influence<0)
	{
		if(M[row * n + col]!=-1)
			*flag = 0;
		newM[row*n + col] = -1;
	}

}

