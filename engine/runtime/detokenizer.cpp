#include "detokenizer.h"

#include <cstdlib>
#include <cstring>

static bool read_all(const char* path, void** out, size_t* bytes)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);

    void* buf = malloc((size_t)n);
    if (!buf) { fclose(f); return false; }

    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return false;
    }
    fclose(f);

    *out = buf;
    *bytes = (size_t)n;
    return true;
}

Detokenizer detok_load(const char* dir, int vocab_size)
{
    Detokenizer d;
    d.vocab_size = vocab_size;
    d.blob = nullptr;
    d.offsets = nullptr;
    d.loaded = false;

    char path[1024];
    void* blob = nullptr;
    void* offs = nullptr;
    size_t blob_bytes = 0, offs_bytes = 0;

    snprintf(path, sizeof(path), "%s/tokenizer/token_blob.bin", dir);
    if (!read_all(path, &blob, &blob_bytes)) return d;

    snprintf(path, sizeof(path), "%s/tokenizer/token_offsets.bin", dir);
    if (!read_all(path, &offs, &offs_bytes)) { free(blob); return d; }

    if (offs_bytes != (size_t)(vocab_size + 1) * sizeof(unsigned int)) {
        fprintf(stderr, "detokenizer: offset table has %zu entries, expected %d\n",
                offs_bytes / sizeof(unsigned int), vocab_size + 1);
        free(blob); free(offs);
        return d;
    }

    d.blob = (unsigned char*)blob;
    d.offsets = (unsigned int*)offs;
    d.loaded = true;
    return d;
}

void detok_free(Detokenizer& d)
{
    free(d.blob);
    free(d.offsets);
    d.blob = nullptr;
    d.offsets = nullptr;
    d.loaded = false;
}

void detok_emit(const Detokenizer& d, int id, FILE* out)
{
    if (!d.loaded) { fprintf(out, "[%d]", id); return; }
    if (id < 0 || id >= d.vocab_size) { fprintf(out, "[?%d]", id); return; }

    unsigned int start = d.offsets[id];
    unsigned int end   = d.offsets[id + 1];
    if (end > start)
        fwrite(d.blob + start, 1, end - start, out);
}
