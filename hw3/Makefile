NVCCFLAGS  = -O3 -std=c++11 -Xptxas=-v -arch=sm_61 -ftz=true -use_fast_math
NVCC     = nvcc 
LDFLAGS = -lz

TARGETS = ok other readout hw3-2

.PHONY: all
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)