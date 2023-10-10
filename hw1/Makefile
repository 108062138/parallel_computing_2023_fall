CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3
CFLAGS = -O3
TARGETS = hw1
SOURCES = hw1.cc
.PHONY: all
all: $(TARGETS)
	$(CC) $(CXXFLAGS) $(SOURCES) -o $(TARGETS) -lm -lstdc++

.PHONY: clean
clean:
	rm -f $(TARGETS)
