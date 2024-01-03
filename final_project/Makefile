CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17
LDFLAGS = -lpng -ljpeg
OMPFLAGS = -fopenmp

# Name of your executables
TARGET_SEQ = dct_seq
TARGET_OMP = dct_omp

# Source files for each executable
SRC_SEQ = dct_seq.cpp
SRC_OMP = dct_omp.cpp

# Object files for each executable
OBJ_SEQ = $(SRC_SEQ:.cpp=.o)
OBJ_OMP = $(SRC_OMP:.cpp=.o)

# Default target
all: $(TARGET_SEQ) $(TARGET_OMP)

# How to link the sequential program
$(TARGET_SEQ): $(OBJ_SEQ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# How to link the OpenMP program
$(TARGET_OMP): $(OBJ_OMP)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)
$(TARGET_SEQ): $(OBJ_SEQ)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)

# This is a rule for generating the .o files from .cpp files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Special rule for OpenMP .o files to include the OpenMP flags
$(OBJ_OMP): $(SRC_OMP)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@
# Special rule for OpenMP .o files to include the OpenMP flags
$(OBJ_SEQ): $(SRC_SEQ)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

# Clean target for cleaning up
clean:
	rm -f $(OBJ_SEQ) $(OBJ_OMP) $(TARGET_SEQ) $(TARGET_OMP)

# PHONY targets are not files
.PHONY: all clean
