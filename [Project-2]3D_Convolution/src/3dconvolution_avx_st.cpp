#include<stdio.h>
#include<stdlib.h>
#include<x86intrin.h>
#include<immintrin.h>
#include<unistd.h>
#include<ctype.h>
#include<inttypes.h>
#include<string.h>
#include<time.h>

__m256i masktable_m256(int k) {
    int v[15] = { -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };
    int j = 7 - k;
    return _mm256_setr_epi32(v[j], v[j + 1], v[j + 2], v[j + 3], v[j + 4], v[j + 5], v[j + 6], v[j + 7]);
}
__m128i masktable_m128(int k) {

    int v[7] = { -1, -1, -1, 0, 0, 0, 0 };
    int j = 3 - k;

    return _mm_setr_epi32(v[j], v[j + 1], v[j + 2], v[j + 3]);
}
__m128 _mm256d_sum(__m256d hi, __m256d lo) {
    __m256d u = _mm256_hadd_pd(lo, hi);
    __m256d v = _mm256_hadd_pd(u, _mm256_setzero_pd());
    __m128d w = _mm_add_pd(_mm256_extractf128_pd(v, 1), _mm256_castpd256_pd128(v));

    return _mm_cvtpd_ps(w);
}

void zero_padding_3d(unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                     unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                     unsigned int pad_size,
                     float* inmap_ptr, float* outmap_ptr) {
    
    for (unsigned int iz = 0; iz < indepth; iz++) {
        /* xyz center */ {
            const unsigned int length = inwidth;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = masktable_m256(k);

            unsigned int inmap_idx = inwidth * inheight * iz;
            unsigned int outmap_idx = (pad_size + outwidth * (pad_size + outheight * (iz + pad_size)));

            for (unsigned int iy = 0; iy < inheight; iy++) {
                for (unsigned int i = 0; i < j; i += 8) {
                    __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }

                inmap_idx += inwidth;
                outmap_idx += outwidth;
            }
        }

        /* x left */ {
            const unsigned int length = pad_size;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = masktable_m256(k);
            const __m256 x = _mm256_setzero_ps();

            unsigned int outmap_idx = outwidth * (pad_size + outheight * (iz + pad_size));

            for (unsigned int iy = 0; iy < inheight; iy++) {
                for (unsigned int i = 0; i < j; i += 8) {
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }

                outmap_idx += outwidth;
            }
        }

        /* x right */ {
            const unsigned int length = pad_size;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = masktable_m256(k);
            const __m256 x = _mm256_setzero_ps();

            unsigned int outmap_idx = (pad_size + inwidth + outwidth * (pad_size + outheight * (iz + pad_size)));

            for (unsigned int iy = 0; iy < inheight; iy++) {
                for (unsigned int i = 0; i < j; i += 8) {
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }

                outmap_idx += outwidth;
            }
        }

        /* y top */ {
            const unsigned int length = outwidth * pad_size;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = masktable_m256(k);
            const __m256 x = _mm256_setzero_ps();

            const unsigned int outmap_idx = outwidth * outheight * (iz + pad_size);

            for (unsigned int i = 0; i < j; i += 8) {
                _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
            }
            if (k > 0) {
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
            }
        }

        /* y bottom */ {
            const unsigned int length = outwidth * pad_size;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = masktable_m256(k);
            const __m256 x = _mm256_setzero_ps();

            const unsigned int outmap_idx = outwidth * (pad_size + inheight + outheight * (iz + pad_size));

            for (unsigned int i = 0; i < j; i += 8) {
                _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
            }
            if (k > 0) {
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
            }
        }
    }

    /*z front*/{
        const unsigned int length = outwidth * outheight * pad_size;
        const unsigned int j = length & ~7u, k = length - j;
        const __m256i mask = masktable_m256(k);
        const __m256 x = _mm256_setzero_ps();

        const unsigned int outmap_idx = 0;

        for (unsigned int i = 0; i < j; i += 8) {
            _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
        }
        if (k > 0) {
            _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
        }
    }

    /*z rear*/ {
        const unsigned int length = outwidth * outheight * pad_size;
        const unsigned int j = length & ~7u, k = length - j;
        const __m256i mask = masktable_m256(k);
        const __m256 x = _mm256_setzero_ps();

        const unsigned int outmap_idx = outwidth * outheight * (pad_size + indepth);

        for (unsigned int i = 0; i < j; i += 8) {
            _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
        }
        if (k > 0) {
            _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
        }
    }
}

void convolution_3d(unsigned int inwidth, unsigned int outwidth, unsigned int ksize,
                    unsigned int inheight, unsigned int outheight, 
                    unsigned int indepth, unsigned int outdepth, 
                    float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {
        
    const __m256i mask = masktable_m256(1);
    const __m128i mask1 = masktable_m128(1);

    for (unsigned int oz = 0; oz < outdepth; oz++) {
        for (unsigned int oy = 0; oy < outheight; oy++) {
            for (unsigned int ox = 0; ox < outwidth; ox++) {
                    __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                    for (unsigned int kz = 0, iz = oz; kz < ksize; kz++, iz++) {
                        for (unsigned int ky = 0, iy = oy; ky < ksize; ky++, iy++) {
                            for (unsigned int kx = 0, ix = ox; kx < ksize; kx++, ix++) { 
                                    __m256 u = _mm256_maskload_ps(inmap_ptr + (ix + inwidth * (iy + inheight * iz)), mask);
                                    __m256 v = _mm256_maskload_ps(kernel_ptr + (kx + ksize * (ky + ksize * kz)), mask);

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                                    uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                            }
                        }
                    }
                    _mm_maskstore_ps(outmap_ptr + (ox + outwidth * (oy + outheight * oz)), mask1, _mm256d_sum(uv_hi, uv_lo));
                
            }
        }
    }
}

int main(int argc, char **argv)
{
	FILE *input_file, *kernel_file, *output_file;
	float *N, *N_padd, *M, *P, *answer;
	int i_depth, i_height, i_width, k_size, o_depth, o_height, o_width;
	ssize_t N_size, P_size;
	bool equal;

	input_file = fopen(argv[1], "r");
	if(input_file == NULL){
		printf("input_file open failed\n");
		exit(0);
	}
	kernel_file = fopen(argv[2], "r");
	if(kernel_file == NULL){
		printf("kernel_file open failed\n");
		exit(0);
	}	
	output_file = fopen(argv[3], "r");
	if(output_file == NULL){
		printf("output_file open failed\n");
		exit(0);
	}

	fscanf(input_file, "%d %d %d", &i_depth, &i_height, &i_width);
	N_size = i_depth * i_height * i_width * sizeof(float);
	N = (float*)malloc(N_size);
	for (int i = 0; i < i_depth * i_height * i_width; i++)
		fscanf(input_file, "%f", &N[i]);
	
	fscanf(kernel_file, "%d", &k_size);
	M = (float*)malloc(k_size * k_size * k_size * sizeof(float));
	for (int i = 0; i < k_size * k_size * k_size; i++)
		fscanf(kernel_file, "%f", &M[i]);	

	fscanf(output_file, "%d %d %d", &o_depth, &o_height, &o_width);
	P_size = o_depth * o_height * o_width * sizeof(float);
	P = (float*)malloc(P_size);
	answer = (float*)malloc(P_size);
	for (int i = 0; i < o_depth * o_height * o_width; i++)
		fscanf(output_file, "%f", &answer[i]);

	unsigned int i_padd_depth = i_depth + (k_size - 1);
	unsigned int i_padd_height = i_height + (k_size - 1);
	unsigned int i_padd_width = i_width + (k_size - 1);
	
	N_padd = (float*)malloc(i_padd_depth * i_padd_height * i_padd_width * sizeof(float));
	zero_padding_3d(
			i_width, i_height, i_depth,
			i_padd_width, i_padd_height, i_padd_depth,
			(k_size - 1) / 2,
			N, N_padd
			);

	unsigned int p_depth = i_padd_depth - (k_size - 1);
	unsigned int p_height = i_padd_height - (k_size - 1);
	unsigned int p_width = i_padd_width - (k_size - 1);

	unsigned long long start, end;
	start = __rdtsc();
	convolution_3d(
			i_padd_width, p_width, k_size,
			i_padd_height, p_height, 
			i_padd_depth, p_depth, 
		       	N_padd, P, M
			);
	end = __rdtsc();

	equal = true;
	for (int i = 0; i < o_depth * o_height * o_width && equal; i++) {
		if (abs(answer[i] - P[i]) >= 0.001f) {
			equal = false;
			printf("NOT EQUAL!\n");

		}
	}

	if (equal) {
		printf("Results are equal!\n");
	}
	else {
		printf("Results are NOT equal!\n");
	}
	printf("Single thread AVX execution Elapsed Time : %fms\n", (double)(end - start)/1197250);
	free(N);
	free(M);
	free(P);
	free(answer);
	fclose(input_file);
	fclose(kernel_file);
	fclose(output_file);
	
	return 0;
}
