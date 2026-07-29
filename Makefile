ARCH    ?= sm_75
TILE_Q  ?= 16
TILE_KV ?= 16
HEAD_DIM?= 64

NVCC    = nvcc
FLAGS   = -O3 -std=c++17 --use_fast_math -arch=$(ARCH) \
          -DTILE_Q=$(TILE_Q) -DTILE_KV=$(TILE_KV) -DHEAD_DIM=$(HEAD_DIM)

SRC_ATTN   = main.cu naive_attention.cu fused_attention.cu
TARGET_ATTN = attention

SRC_DAY1    = test_day1.cu fused_attention_decode.cu elementwise.cu
TARGET_DAY1 = test_day1

all: $(TARGET_ATTN) $(TARGET_DAY1)

$(TARGET_ATTN): $(SRC_ATTN) attention.cuh
	$(NVCC) $(FLAGS) -o $@ $(SRC_ATTN)

$(TARGET_DAY1): $(SRC_DAY1) attention.cuh elementwise.cuh
	$(NVCC) $(FLAGS) -o $@ $(SRC_DAY1)

clean:
	rm -f $(TARGET_ATTN) $(TARGET_DAY1)

.PHONY: all clean
