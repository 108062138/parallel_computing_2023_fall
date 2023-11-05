CC = gcc
CXX = g++
LDLIBS = -lpng #-march=native -fopt-info-vec-optimized
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp

custom_mpi: CC = mpicc
custom_mpi: CXX = mpicxx
custom_mpi: CFLAGS += -fopenmp

CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a thread_pools hw2b custom_mpi

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)