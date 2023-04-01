#pragma once

#include <string>
#include <map>
#include <vector>
#include "ggml.h"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

#define MAGIC_VERSION_NUMBER 0x67676d6c

// determine number of model parts based on the dimension
static const std::map<int, int> LLAMA_N_PARTS = {
    { 4096, 1 },
    { 5120, 1 },
    { 6656, 1 },
    { 8192, 1 },
};

// default hparams (LLaMA 7B)
struct llama_hparams {
    int32_t n_vocab = 32001;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

struct llama_layer {
    // normalization
    struct ggml_tensor * attention_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;

    // normalization
    struct ggml_tensor * ffn_norm;

    // ff
    struct ggml_tensor * w1;
    struct ggml_tensor * w2;
    struct ggml_tensor * w3;
};

struct llama_model {
    llama_hparams hparams;

    struct ggml_tensor * tok_embeddings;

    struct ggml_tensor * norm;
    struct ggml_tensor * output;

    std::vector<llama_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};
