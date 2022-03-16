#include <x86intrin.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdbool.h>
#include <unistd.h>

float answer[4][4] = {
	{10, 20, 30, 40},
	{50, 100, 150, 200},
	{90, 180, 270, 360},
	{130, 260, 390, 520}
};

__m256 multiply_simple(int i, float** a, __m256* matrix)
{

        __m256 ymm0;
        __m256 ymm1;
        __m256 ymm2;
        __m256 ymm3;
	__m256 result;

        ymm0 = _mm256_broadcast_ss(&a[i][0]);
        ymm1 = _mm256_broadcast_ss(&a[i][1]);
        ymm2 = _mm256_broadcast_ss(&a[i][2]);
        ymm3 = _mm256_broadcast_ss(&a[i][3]);
        ymm0 = _mm256_mul_ps(ymm0, matrix[0]);
        ymm1 = _mm256_mul_ps(ymm1, matrix[1]);
        ymm0 = _mm256_add_ps(ymm0, ymm1);
        ymm2 = _mm256_mul_ps(ymm2, matrix[2]);
        ymm3 = _mm256_mul_ps(ymm3, matrix[3]);
        ymm2 = _mm256_add_ps(ymm2, ymm3);
	result = _mm256_add_ps(ymm0, ymm2);

	return(result);
}

int main(int argc, char *argv[]){

	long unsigned int start, end;
        int m = 4, n = 4;
	
	int opt, mode;
	while((opt = getopt(argc, argv, "v:")) != -1)
	{
		if(opt == 'v')
			mode = atoi(optarg);
	}

	__m256* matrix;
	float** a = malloc(sizeof(float*) * 4);
	float** b = malloc(sizeof(float*) * 4);

	for(int i = 0 ; i < 4 ; i++){
		a[i] = aligned_alloc(32, sizeof(float) * 4);
		b[i] = aligned_alloc(32, sizeof(float) * 4);
	}

	float* temp = aligned_alloc(32, sizeof(float) * 4);
	float* temp1 = aligned_alloc(32, sizeof(float) * 4);
	matrix = aligned_alloc(32, sizeof(__m256) * 4);

	for (int i = 0 ; i < 4 ; i++){
		for(int j = 0 ; j < 4 ; j++){
			temp[j] = i*10 + j + 1;
			temp1[j] = j+1;
		}
		matrix[i] = _mm256_load_ps(temp);
		_mm256_store_ps(a[i], matrix[i]);
		matrix[i] = _mm256_load_ps(temp1);
		_mm256_store_ps(b[i], matrix[i]);
	}

	float *ptr = aligned_alloc(32, sizeof(float) * n);
	
	//Non-AVX
	
	if(mode == 0){
	
	
	float result[4][4] = {};

	start = __rdtsc();
	for(int i = 0; i < 4 ; i++){
		for(int j = 0 ; j < 4 ; j++){
			for(int k = 0 ; k < 4; k++){
				result[i][j] += a[i][k] * b[k][j];
				//printf("%d, %d: %f %f %f\n",i, j,  result[i][j], a[i][k], b[k][k]);
			}
		}
	}
	end = __rdtsc();
	
        printf("Scalar: Elapsed time with Non-AVX: %" PRIu64 "\n", end  -start);
	
        bool correct = true;
        for(int i = 0 ; i < 4 ; i++ ){
                for(int j = 0 ; j < 4 ; j++){
                        if(result[i][j] != answer[i][j]){
                                correct = false;
                        }
                }
        }

        if(correct){
                printf("Scalar: Matrix multiplication is correct\n");
        }else{
                 printf("Scalarr: Matrix multiplication is not correct\n");
        }

        }
	
	if(mode == 1){
	
	__m256* result2 = aligned_alloc(32, sizeof(__m256) * 4);

	//AVX

	start = __rdtsc();
	result2[0] = multiply_simple(0, a, matrix);
	result2[1] = multiply_simple(1, a, matrix);
	result2[2] = multiply_simple(2, a, matrix);
	result2[3] = multiply_simple(3, a, matrix);
	end = __rdtsc();
        printf("Vector: Elapsed time with AVX: %" PRIu64 "\n", end  -start);
	

	bool correct = true;
	for(int i = 0 ; i < 4 ; i++ ){
		_mm256_store_ps(ptr, result2[i]);
		for(int j = 0 ; j < 4 ; j++){
			if(answer[i][j] != ptr[j]){
				correct = false;
			}
		}
	}

	if(correct){
		printf("Vector: Matrix multiplication is correct\n");
	}else{
		 printf("Vector: Matrix multiplication is not correct\n");
	}

	}

}
