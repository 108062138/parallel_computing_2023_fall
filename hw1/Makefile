CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -lm -lstdc++ # -L/opt/ipm/lib -lipm
CFLAGS = -O3
TARGETS = hw1
# SOURCES = hw1.cc
SOURCES = modify.cc
.PHONY: all
all: $(TARGETS)
	$(CC) $(CXXFLAGS) $(SOURCES) -o $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
