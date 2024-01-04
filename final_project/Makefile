# Compiler settings
CXX = g++
MPICXX = mpicxx  # MPI compiler wrapper
CXXFLAGS = -Wall -Wextra -std=c++17
LDFLAGS = -lpng -ljpeg
OMPFLAGS = -fopenmp

# Name of your executables
# TARGET_SEQ = dct_seq
TARGET_OMP = dct_omp
TARGET_OMP_MPI = dct_omp_mpi  # New target for OpenMP + MPI
TARGET_OMP_MPI_VEC = dct_omp_mpi_vec  # New target for OpenMP + MPI + vectorization

# Source files for each executable
# SRC_SEQ = dct_seq.cpp
SRC_OMP = dct_omp.cpp
SRC_OMP_MPI = dct_omp_mpi.cpp  # Source for the OpenMP + MPI program
SRC_OMP_MPI_VEC = dct_omp_mpi_vec.cpp  # Source for the OpenMP + MPI + vectorization program

# Object files for each executable
# OBJ_SEQ = $(SRC_SEQ:.cpp=.o)
OBJ_OMP = $(SRC_OMP:.cpp=.o)
OBJ_OMP_MPI = $(SRC_OMP_MPI:.cpp=.o)  # Object file for OpenMP + MPI program
OBJ_OMP_MPI_VEC = $(SRC_OMP_MPI_VEC:.cpp=.o)  # Object file for OpenMP + MPI + vectorization program

# Default target
all: $(TARGET_OMP) $(TARGET_OMP_MPI) $(TARGET_OMP_MPI_VEC)

# How to link the sequential program
#$(TARGET_SEQ): $(OBJ_SEQ)
#	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# How to link the OpenMP program
$(TARGET_OMP): $(OBJ_OMP)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)

# How to link the OpenMP + MPI program
$(TARGET_OMP_MPI): $(OBJ_OMP_MPI)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)

# How to link the OpenMP + MPI + vectorization program
$(TARGET_OMP_MPI_VEC): $(OBJ_OMP_MPI_VEC)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)

# This is a rule for generating the .o files from .cpp files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Special rule for OpenMP .o files to include the OpenMP flags
$(OBJ_OMP): $(SRC_OMP)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

# Special rule for OpenMP + MPI .o files
$(OBJ_OMP_MPI): $(SRC_OMP_MPI)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

# Special rule for OpenMP + MPI + vectorization .o files
$(OBJ_OMP_MPI_VEC): $(SRC_OMP_MPI_VEC)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

# Clean target for cleaning up
clean:
	rm -f $(OBJ_SEQ) $(OBJ_OMP) $(OBJ_OMP_MPI) $(TARGET_SEQ) $(TARGET_OMP) $(TARGET_OMP_MPI) $(TARGET_OMP_MPI_VEC)

# PHONY targets are not files
.PHONY: all clean
