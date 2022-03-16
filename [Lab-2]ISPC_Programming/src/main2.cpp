#include "min_max_ispc.h"
#include <x86intrin.h>
#include <stdio.h>
#include <inttypes.h>
#include <cstdlib>
#include <ctime>

int main(int  argc, void**  argv) { 
	
	srand((unsigned int)time(NULL));

	int N = 1024; 
	int* x = new int[N]; 
	int* xmin = new int;
	int* xmax = new int;

	//initialize x here
	for(int i = 0 ; i < N ; i++){
		x[i] = rand()%1000 + 1;
	}

	// execute ISPC code
	ispc::min_ispc(N, x, xmin); 
	ispc::max_ispc(N, x, xmax);
	//printf("ISPC code Time: %" PRIu64 "\n", end  - start);
	
	printf("\n");
	printf("Minimum x: %d\n", *xmin);
	printf("Maximum x: %d\n", *xmax);
	printf("\n");

	return 0; 
} 
