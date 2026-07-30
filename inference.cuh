#pragma once

#include "attention.cuh"
#include "elementwise.cuh"
#include "linear.cuh"
#include "model_types.h"   // ModelConfig / LayerWeights / ModelWeights

// The struct definitions live in model_types.h so the scalar CPU reference and
// the CPU test harness can be compiled with plain g++, no CUDA toolkit and no
// GPU. Host and device weights use the same types; only the pointers differ.

// Load all weights from a directory of .bin files exported by export_weights.py.
// Stages through host memory, then uploads. Allocates GPU memory.
// Call free_model_weights() when done.
ModelWeights load_model_weights(const char* dir);
void free_model_weights(ModelWeights& m);
