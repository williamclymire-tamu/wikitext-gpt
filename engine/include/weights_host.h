#pragma once

#include "model_types.h"
#include <cstddef>

// Host-side loading of an export/ directory produced by export_weights.py.
// No CUDA: this is what the CPU reference uses directly, and what weights.cu
// stages through before uploading to the device.

ModelConfig load_config(const char* dir);

// Reads every .bin into malloc'd host memory. Exits on any missing or
// short-read file -- a silently truncated weight file is the worst possible
// failure mode, since the model still "runs" and just produces garbage.
ModelWeights load_host_weights(const char* dir);

void free_host_weights(ModelWeights& m);

// Sized fp32 read. Fails loudly if the file is not exactly num_floats long.
float* read_f32(const char* path, size_t num_floats);

// Reads int32 token ids; *count is set to the number read.
int* read_i32(const char* path, int* count);
