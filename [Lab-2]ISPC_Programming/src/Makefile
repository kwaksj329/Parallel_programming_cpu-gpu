CC=g++
TARGET=sinx

clean:
	rm sinx

.PHONY: inter-ispc
inter-ispc:
	ispc sinx.ispc -o sinx_ispc.o
	ispc sinx.ispc -h sinx_ispc.h
	g++ -c main.cpp -o main.o
	g++ main.o sinx_ispc.o -o sinx
	./sinx

.PHONY: blocked-ispc
blocked-ispc:
	ispc sinx_blocked.ispc -o sinx_ispc.o
	ispc sinx_blocked.ispc -h sinx_ispc.h
	g++ -c main.cpp -o main.o
	g++ main.o sinx_ispc.o -o sinx
	./sinx

.PHONY: min-max
min-max:
	ispc min_max.ispc -o min_max_ispc.o
	ispc min_max.ispc -h min_max_ispc.h
	g++ -c main2.cpp -o main2.o
	g++ main2.o min_max_ispc.o -o sinx2
	./sinx2
