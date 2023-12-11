CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -lm -pthread -fopenmp
CFLAGS = -O3
NVCCFLAGS = -O3 -std=c++11 -Xptxas=-v -arch=sm_61 -use_fast_math
NVCC = nvcc
LDFLAGS = -lz

TARGETS = hw3-2 hw3-3
SOURCES = hw3-1.cc

.PHONY: all
all: $(TARGETS)

lab2_pthread: lab2_pthread.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)