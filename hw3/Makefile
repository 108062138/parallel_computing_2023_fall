CC = mpicc
CXX = g++
CXXFLAGS = -std=c++17 -O3 -lm -fopenmp
CFLAGS = -O3
NVCCFLAGS = -O3 -std=c++11 -Xptxas=-v -arch=sm_61 -use_fast_math -Xcompiler -fopenmp
NVCC = nvcc
LDFLAGS = -lz

TARGETS = hw3-2 hw3-3 #hw3-2-prof
SOURCES = hw3-1.cc

.PHONY: all
all: $(TARGETS)

lab2_pthread: hw3-1.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)