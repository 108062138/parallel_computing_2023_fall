CC = gcc
CXX = clang++
LDLIBS = -lpng #-march=native -fopt-info-vec-optimized
CFLAGS = -lm -O3 #-L/opt/ipm/lib -fno-math-errno -funsafe-math-optimizations -ffinite-math-only -fno-trapping-math -freciprocal-math
hw2a: CFLAGS += -pthread -lrt
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp

custom_mpi: CC = mpicc
custom_mpi: CXX = mpicxx
custom_mpi: CFLAGS += -fopenmp

CXXFLAGS = $(CFLAGS)
TARGETS =  hw2a hw2b hw2seq
.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)