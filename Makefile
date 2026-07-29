ARCH    ?= sm_75
TILE_Q  ?= 16
TILE_KV ?= 16
HEAD_DIM?= 64

NVCC    = nvcc
FLAGS   = -O3 -std=c++17 --use_fast_math -arch=$(ARCH) \
          -DTILE_Q=$(TILE_Q) -DTILE_KV=$(TILE_KV) -DHEAD_DIM=$(HEAD_DIM)

SRC     = main.cu naive_attention.cu fused_attention.cu
TARGET  = attention

all: $(TARGET)

$(TARGET): $(SRC) attention.cuh
	$(NVCC) $(FLAGS) -o $@ $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean
