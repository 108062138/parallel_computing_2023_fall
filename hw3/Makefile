CC = gcc
CXX = clang++
LDLIBS = -lpng
CFLAGS = -lm -O3
floyd_warshell: CFLAGS += -pthread -lrt

CXXFLAGS = $(CFLAGS)
TARGETS =  floyd_warshell
.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)