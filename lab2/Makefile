CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -lm -pthread -fopenmp -lstdc++ # -L/opt/ipm/lib -lipm
CFLAGS = -O3
TARGETS = lab2_pthread lab2_omp
# SOURCES = hw1.cc
SOURCES = lab2_pthread.cc lab2_omp.cc
.PHONY: all
all: $(TARGETS)
	$(CC) $(CXXFLAGS) $(SOURCES) -o $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
