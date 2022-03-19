#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#define TILE_SIZE 3
#define KERNEL_SIZE 3
#define BLOCK_SIZE (TILE_SIZE + (KERNEL_SIZE -1))

__constant__ float Mc[KERNEL_SIZE][KERNEL_SIZE];

__global__ void convolution(float *N, float *P, int Nh, int Nw, int Ph, int Pw ){

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;

	int row_i = row_o - ((KERNEL_SIZE - 1) / 2);
	int col_i = col_o - ((KERNEL_SIZE - 1) / 2);

	float output = 0.0f;
	__shared__ float Ns[TILE_SIZE + KERNEL_SIZE -1][TILE_SIZE + KERNEL_SIZE -1];

	if((row_i >= 0) && (row_i < Nh) && (col_i >= 0) && (col_i < Nw))
		Ns[ty][tx] = N[row_i * Nw + col_i];
	else
		Ns[ty][tx] = 0.0f;

	if(ty < TILE_SIZE && tx < TILE_SIZE){
		for(int i = 0 ; i < KERNEL_SIZE ; i++ ){
			for(int j = 0 ; j < KERNEL_SIZE ; j++){
				output += Mc[i][j] * Ns[i+ty][j+tx];
			}
		}
	

		if((row_o < Ph) && (col_o < Pw))
			P[row_o * Pw + col_o] = output;
	}
}

void verification(float *N, float *M, const float *P, int Nh, int Nw,  int Rows, int Columns) {
	int r, c, h, w;
	int row_i, col_i;
	bool equal;
	float* results;

	results = (float*)malloc(Rows * Columns * sizeof(float));
	memset(results, 0, Rows * Columns * sizeof(float));

	for (r = 0; r < Rows; r++) {
		for (c = 0; c < Columns; c++) {
			for (h = 0; h < KERNEL_SIZE; h++) {
				for (w = 0; w < KERNEL_SIZE; w++) {
					row_i = r - ((KERNEL_SIZE - 1) / 2) + h;
					col_i = c - ((KERNEL_SIZE - 1) / 2) + w;
					if ((row_i >= 0) && (row_i < Nh) && (col_i >= 0) && (col_i < Nw)) {
						results[r*Columns + c] += (M[h*KERNEL_SIZE + w] * N[row_i*Nw + col_i]);
					}
				}
			}
		}
	}

	equal = true;
	for (int i = 0; i < Rows * Columns && equal; i++) {
		if (abs(results[i] - P[i]) >= 0.001f) {
			equal = false;
			printf("NOT EQUAL!\n");
			//printf("%d %f %f\n",i,  results[i] , P[i]);
		}
	}

	if (equal) {
		printf("Results are equal!\n");
	}
	else {
		printf("Results are NOT equal!\n");
	}

	free(results);
	return;
}

int main(){

	int row, col, Ph, Pw;

	printf("Row Col: \n");
	scanf("%d %d", &row, &col);

	Ph = row - KERNEL_SIZE + 1;
	Pw = col - KERNEL_SIZE + 1;
	
	float *N, *P, *DN, *M, *DP;
	N = (float*)malloc(sizeof(float) * row * col);
	M = (float*)malloc(sizeof(float) * KERNEL_SIZE * KERNEL_SIZE);
	P = (float*)malloc(sizeof(float) * Ph * Pw);
	cudaMalloc((void**)&DN, sizeof(float) * row * col);
	cudaMalloc((void**)&DP, sizeof(float) * Ph * Pw);
	memset(N, 0, sizeof(float) * row * col);
	memset(M, 0, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE);
	memset(P, 0, sizeof(float) * Ph * Pw);
	
	for(int i = 0 ; i < row * col ; i++){
		N[i] = (float)(rand() % 50) / 10;
	
	}

	for(int i = 0 ; i < KERNEL_SIZE * KERNEL_SIZE ; i++){
		M[i] = (float)(rand() % 50) / 10;
	}
	
	cudaMemcpy(DN, N, sizeof(float) * row * col, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Mc, M, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE);
	cudaMemcpy(DP, P, sizeof(float) * Ph * Pw, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(ceil((float)Pw / TILE_SIZE), ceil((float)Ph / TILE_SIZE));
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	convolution<<<dimGrid, dimBlock>>>(DN, DP, row, col, Ph, Pw);

	cudaMemcpy(P, DP, sizeof(float) * Ph * Pw, cudaMemcpyDeviceToHost);

	verification(N, M, P, row, col, Ph, Pw);

	cudaFree(DN);
	cudaFree(DP);
	free(N);
	free(M);
	free(P);

	return 0;
	
}
