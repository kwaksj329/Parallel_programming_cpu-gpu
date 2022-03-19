CC=nvcc

.PHONY: nontile
nontile:
	nvcc non_tiled.cu -o nontile	

.PHONY: tile
tile: 
	nvcc tiled.cu -o tile

clean:
	rm nontile
	rm tile

