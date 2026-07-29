# Fused Attention CUDA Kernel — Build System
#
# Usage:
#   make            # build the test/bench binary
#   make test       # generate reference data + run correctness tests
#   make bench      # run benchmark sweep
#   make profile    # ncu profiling (requires admin permissions)
#   make clean

NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++17 --use_fast_math
# Add your GPU's compute capability here. Common values:
#   sm_75  (T4, RTX 2080)
#   sm_80  (A100)
#   sm_86  (RTX 3090, A40)
#   sm_89  (RTX 4090, L40)
#   sm_90  (H100)
ARCH      ?= sm_80
NVCCFLAGS += -arch=$(ARCH)

# Tile sizes (override on command line: make TILE_Q=32 TILE_KV=32)
TILE_Q    ?= 16
TILE_KV   ?= 16
HEAD_DIM  ?= 64
NVCCFLAGS += -DTILE_Q=$(TILE_Q) -DTILE_KV=$(TILE_KV) -DHEAD_DIM=$(HEAD_DIM)

SRCS := main.cu naive_attention.cu fused_attention.cu
TARGET := attention

.PHONY: all test bench profile clean

all: $(TARGET)

$(TARGET): $(SRCS) attention.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

test_data/meta.json:
	python3 generate_reference.py

test: $(TARGET) test_data/meta.json
	./$(TARGET) test test_data

bench: $(TARGET)
	./$(TARGET) bench

profile: $(TARGET)
	ncu --set full -o profile_report ./$(TARGET) bench

clean:
	rm -f $(TARGET) profile_report.ncu-rep
	rm -rf test_data
