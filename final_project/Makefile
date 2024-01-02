CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11
LDFLAGS = -lpng -ljpeg

# Name of your executable
TARGET = dct_seq

# List of your source files
SRC = dct_seq.cpp

# Object files are the .cpp files, but with .o extensions
OBJ = $(SRC:.cpp=.o)

# Default target
all: $(TARGET)

# How to link the program
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# This is a rule for generating the .o files from .cpp files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target for cleaning up
clean:
	rm -f $(OBJ) $(TARGET)

# PHONY targets are not files
.PHONY: all clean
