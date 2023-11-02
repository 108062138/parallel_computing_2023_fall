CC = gcc
CXX = g++
LDLIBS = -lpng #-march=native -fopt-info-vec-optimized
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a thread_pools

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
