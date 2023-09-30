CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3
CFLAGS = -O3
TARGETS = lab1
SOURCES = lab1.cc
.PHONY: all
all: $(TARGETS)
	$(CC) $(CXXFLAGS) $(SOURCES) -o $(TARGETS) -lm

.PHONY: clean
clean:
	rm -f $(TARGETS)
