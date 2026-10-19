# Built on the gateway (CUDA 12.9 supports sm_70).
NVCC ?= /usr/local/cuda-12.9/bin/nvcc
bakeoff: src/bakeoff.cu
	$(NVCC) -O3 -std=c++17 -arch=sm_70 -lineinfo -o $@ $< -lcublas
clean:
	rm -f bakeoff
