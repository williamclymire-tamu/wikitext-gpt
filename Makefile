ARCH    ?= sm_75
TILE_Q  ?= 16
TILE_KV ?= 16
HEAD_DIM?= 64

NVCC    = nvcc
FLAGS   = -O3 -std=c++17 --use_fast_math -arch=$(ARCH) \
          -DTILE_Q=$(TILE_Q) -DTILE_KV=$(TILE_KV) -DHEAD_DIM=$(HEAD_DIM)

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17

SRC_ATTN   = main.cu naive_attention.cu fused_attention.cu
TARGET_ATTN = attention

SRC_DAY1    = test_day1.cu fused_attention_decode.cu elementwise.cu
TARGET_DAY1 = test_day1

# Full inference engine: weight loading, forward pass, KV cache, sampling.
SRC_ENGINE = run.cu transformer.cu weights.cu weights_host.cpp \
             fused_attention.cu fused_attention_decode.cu \
             elementwise.cu linear.cu detokenizer.cpp
TARGET_ENGINE = engine

# CPU-only reference + harness. Builds with plain g++, no toolkit and no GPU,
# so the forward-pass wiring can be validated anywhere.
SRC_CPU    = test_cpu.cpp reference.cpp weights_host.cpp
TARGET_CPU = test_cpu

all: $(TARGET_ATTN) $(TARGET_DAY1) $(TARGET_ENGINE)

$(TARGET_ATTN): $(SRC_ATTN) attention.cuh
	$(NVCC) $(FLAGS) -o $@ $(SRC_ATTN)

$(TARGET_DAY1): $(SRC_DAY1) attention.cuh elementwise.cuh
	$(NVCC) $(FLAGS) -o $@ $(SRC_DAY1)

$(TARGET_ENGINE): $(SRC_ENGINE) attention.cuh elementwise.cuh linear.cuh \
                  inference.cuh transformer.cuh model_types.h weights_host.h \
                  detokenizer.h checks.h
	$(NVCC) $(FLAGS) -o $@ $(SRC_ENGINE) -lcublas

$(TARGET_CPU): $(SRC_CPU) model_types.h weights_host.h reference.h checks.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_CPU)

clean:
	rm -f $(TARGET_ATTN) $(TARGET_DAY1) $(TARGET_ENGINE) $(TARGET_CPU)

.PHONY: all clean
