#pragma once

#include <cstdio>

// Decode-only tokenizer.
//
// export_weights.py already inverted GPT-2's byte<->unicode mapping and wrote a
// flat table of raw bytes per token id, so decoding here is a blob lookup and a
// write. No unicode tables, no merge rules, no UTF-8 handling in C++ at all --
// multi-byte characters fall out correctly because the bytes were never decoded
// in the first place.
//
// Encoding stays in Python. A C++ BPE encoder is a day of work and buys nothing
// the engine needs; the driver takes token ids on stdin or an --ids argument.

struct Detokenizer {
    int vocab_size;
    unsigned char* blob;
    unsigned int*  offsets;   // [vocab_size + 1]
    bool loaded;
};

// Looks for <dir>/tokenizer/{token_blob.bin,token_offsets.bin}. Returns a
// Detokenizer with loaded=false if they are absent, in which case the driver
// prints numeric ids instead of failing.
Detokenizer detok_load(const char* dir, int vocab_size);
void detok_free(Detokenizer& d);

// Writes the raw bytes for one token. Nothing is buffered or interpreted, so
// output can be piped straight to a terminal or a file.
void detok_emit(const Detokenizer& d, int id, FILE* out);
