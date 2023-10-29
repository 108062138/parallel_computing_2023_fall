CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -lm -pthread -fopenmp
CFLAGS = -O3
TARGETS = lab2_pthread lab2_omp lab2_hybrid
SOURCES = lab2_pthread.cc lab2_omp.cc lab2_hybrid.cc

.PHONY: all
all: $(TARGETS)

lab2_pthread: lab2_pthread.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

lab2_omp: lab2_omp.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

lab2_hybrid: lab2_hybrid.cc
	$(CXX) $(CXXFLAGS) $^ -o $@


.PHONY: clean
clean:
	rm -f $(TARGETS)
