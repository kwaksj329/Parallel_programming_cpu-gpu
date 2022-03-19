#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <cmath>
#define TILE_WIDTH 16


__global__
void MatrixMulKernel(float* Adevice, float* Bdevice, float* Cdevice, int Width) {

	int Row = blockIdx.y * blockDim.y + threadIdx.y;
        int Col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if((Row < Width) && (Col < Width)){
		float Pvalue = 0;
		for(int k = 0 ; k < Width ; k++)
			Pvalue += Adevice[Row * Width + k] * Bdevice[k * Width + Col];
		Cdevice[ Row * Width + Col ] = Pvalue;
	}

}

/*
__global__
void MatrixMulKernel(float* Adevice, float* Bdevice, float* Cdevice, int Width) {

	__shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
	__shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

        int Row = blockIdx.y * blockDim.y + threadIdx.y;
        int Col = blockIdx.x * blockDim.x + threadIdx.x;

	int i = 0;
	int j = 0;
	float temp = 0;

	for(int tile = 0 ; tile < gridDim.x ; tile++){
		
		j = tile * TILE_WIDTH + threadIdx.x;
		i = tile * TILE_WIDTH + threadIdx.y;

		A_shared[threadIdx.y][threadIdx.x] = Adevice[ Row * Width + j ];
		B_shared[threadIdx.y][threadIdx.x] = Bdevice[ i * Width + Col ];

		__syncthreads();

		for(int n = 0 ; n < TILE_WIDTH ; n++){
			temp += A_shared[threadIdx.y][n] * B_shared[n][threadIdx.x];
		}

		__syncthreads();
	}

	Cdevice[ Row * Width + Col ] = temp;
}
*/

int main() {

	int Ax, Ay, Bx, By, Width;
	float *Ahost, *Adevice, *Bhost, *Bdevice, *Chost, *Cdevice, *Answer;
	size_t vector_size;
	float time_ms = 0;

	printf("< Non-tiled kernel test>\n");

	//scan Ax, Ay, Bx, By
	printf("Ax Ay Bx By: \n");
	scanf("%d %d %d %d", &Ax, &Ay, &Bx, &By);
	
	//determine Width & vector_size

	int maxX, maxY, temp;

	maxX = Ax > Bx ? Ax : Bx;
	maxY = Ay > By ? Ay : By;
	Width = maxX > maxY ? maxX : maxY;
	temp = Width / TILE_WIDTH + (Width % TILE_WIDTH == 0 ? 0 : 1);
	Width = temp * TILE_WIDTH;
	
	vector_size = sizeof(float) * Width * Width;

	Ahost = (float*)malloc(vector_size);
	Bhost = (float*)malloc(vector_size);

	memset(Ahost, 0, vector_size);
	memset(Bhost, 0, vector_size);
	

	//set random numbers in A & B matrix

	for(int i = 0 ; i < Ax ; i++){
		for(int j = 0 ; j < Ay ; j++){
			Ahost[Width * i + j] = (float)(rand() % 50) / 10; 
		}
	}

        for(int i = 0 ; i < Bx ; i++){
                for(int j = 0 ; j < By ; j++){
                        Bhost[Width * i + j] = (float)(rand() % 50) / 10;
                }
        }

	Chost = (float*)malloc(vector_size);
	Answer = (float*)malloc(vector_size);
	cudaMalloc((void**)&Adevice, vector_size);
	cudaMalloc((void**)&Bdevice, vector_size);
	cudaMalloc((void**)&Cdevice, vector_size);

	cudaMemcpy(Adevice, Ahost, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(Bdevice, Bhost, vector_size, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(Width/(TILE_WIDTH * 1.0)), ceil(Width/(TILE_WIDTH * 1.0)), 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	cudaEvent_t t1, t2;
	cudaEventCreate(&t1);
	cudaEventCreate(&t2);

	cudaEventRecord(t1, 0);

	MatrixMulKernel<<<dimGrid, dimBlock>>>(Adevice, Bdevice, Cdevice, Width);

	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);
	
	cudaEventElapsedTime(&time_ms, t1, t2);
	cudaEventDestroy(t1);
	cudaEventDestroy(t2);
	
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		// somthing's gone wrong
		// print out the CUDA error as a string
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));

		// we can't recover from the error -- exit the program
		return 1;
	}

	cudaMemcpy(Chost, Cdevice, vector_size, cudaMemcpyDeviceToHost);
	/*
	//in CPU - matrix multiplication
	for(int i = 0 ; i < Width ; ++i){
		for(int j = 0 ; j < Width ; ++j){
			float tnum = 0;
			for(int k = 0 ; k < Width ; ++k){
				tnum += Ahost[Width * i + k] * Bhost[Width * k + j];
			}
			Answer[ Width * i + j ] = tnum;
		}
	}
	*/
	printf("Execution time for non-tiled: %.3f ms\n", time_ms);
	/*
	//check computation result
	bool correct = true;
	for(int i = 0 ; i < Ax ; i++){
		for(int j = 0 ; j < By ; j++){
			if(abs( Answer[i * Width + j] - Chost[i * Width + j] > 0.001)){
				printf("%f %f\n", Answer[i * Width + j], Chost[i * Width + j]);
				correct = false;
			}
		}
	}

	if(correct){
		std::cout<<"Results are equal."<<std::endl;
	}else{
		std::cout<<"Results are not equal."<<std::endl;
	}
	*/

	cudaFree(Adevice);
	cudaFree(Bdevice);
	cudaFree(Cdevice);
	free(Ahost);
	free(Bhost);
	free(Chost);

	return 0;

}

