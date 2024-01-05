CC = mpicc
CXX = g++
CXXFLAGS = -std=c++17 -O3 -lm -fopenmp
CFLAGS = -O3
NVCCFLAGS = -O3 -std=c++11 -Xptxas=-v -arch=sm_61 -use_fast_math -Xcompiler -fopenmp
NVCC = nvcc
LDFLAGS = -lz -lm -lpng -ljpeg

TARGETS = dct_cuda dct_cuda_many_gpu

.PHONY: all
all: $(TARGETS)

lab2_pthread: hw3-1.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)