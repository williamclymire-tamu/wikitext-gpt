ARCH    ?= sm_75
TILE_Q  ?= 16
TILE_KV ?= 16
HEAD_DIM?= 64

NVCC    = nvcc
FLAGS   = -O3 -std=c++17 --use_fast_math -arch=$(ARCH) \
          -DTILE_Q=$(TILE_Q) -DTILE_KV=$(TILE_KV) -DHEAD_DIM=$(HEAD_DIM) \
          -Iinclude

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Iinclude

K       = kernels
R       = runtime
I       = include
TESTS   = ../tests

# ── Standalone kernel tests (unified binary) ─────────────────

SRC_KERNELS  = $(TESTS)/test_kernels.cu \
               $(K)/naive_attention.cu $(K)/fused_attention.cu \
               $(K)/fused_attention_decode.cu $(K)/elementwise.cu
TARGET_KERNELS = $(TESTS)/test_kernels

# ── Full inference engine ─────────────────────────────────────

SRC_ENGINE = run.cu \
             $(R)/transformer.cu $(R)/weights.cu $(R)/weights_host.cpp \
             $(K)/fused_attention.cu $(K)/fused_attention_decode.cu \
             $(K)/elementwise.cu $(K)/linear.cu $(R)/detokenizer.cpp
TARGET_ENGINE = engine

# ── CPU-only reference + harness (no GPU, no toolkit) ─────────

SRC_CPU    = $(TESTS)/test_cpu.cpp $(R)/reference.cpp $(R)/weights_host.cpp
TARGET_CPU = $(TESTS)/test_cpu

# ──────────────────────────────────────────────────────────────

all: $(TARGET_KERNELS) $(TARGET_ENGINE)

$(TARGET_KERNELS): $(SRC_KERNELS) $(I)/attention.cuh $(I)/elementwise.cuh $(I)/checks.h
	$(NVCC) $(FLAGS) -o $@ $(SRC_KERNELS)

$(TARGET_ENGINE): $(SRC_ENGINE) $(I)/attention.cuh $(I)/elementwise.cuh $(I)/linear.cuh \
                  $(I)/inference.cuh $(I)/transformer.cuh $(I)/model_types.h \
                  $(I)/weights_host.h $(I)/detokenizer.h $(I)/checks.h
	$(NVCC) $(FLAGS) -o $@ $(SRC_ENGINE) -lcublas

$(TARGET_CPU): $(SRC_CPU) $(I)/model_types.h $(I)/weights_host.h $(I)/reference.h $(I)/checks.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_CPU)

clean:
	rm -f $(TARGET_KERNELS) $(TARGET_ENGINE) $(TARGET_CPU)

.PHONY: all clean
