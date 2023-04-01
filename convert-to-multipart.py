import io
import os
import shutil
import sys
from argparse import ArgumentParser

# ==============================================================================

EXPECTED_MAGIC = 0x67676d6c
ENDIAN = 'little'
SIZE_U32 = 4
SIZE_I32 = 4
# Context size. llama lets you configure this at runtime, with this default, but
# we'll just hardcode it here for now.
N_CTX = 2048 

# ==============================================================================
# =================================== UTILS ===================================
# ==============================================================================

def read_i32(fin):
    return int.from_bytes(fin.read(SIZE_I32), ENDIAN)
def read_u32(fin):
    return int.from_bytes(fin.read(SIZE_U32), ENDIAN, signed=False)

def pipe_i32(fin: io.BufferedReader,
             fout: io.BufferedWriter,
             new_val=None,
             endian=ENDIAN):
    assert fin is not None
    assert fout is not None

    val = int.from_bytes(fin.read(SIZE_I32), endian)
    to_write = new_val if new_val is not None else val
    fout.write(to_write.to_bytes(SIZE_I32, endian))

    return to_write

def pipe_u32(fin: io.BufferedReader,
             fout: io.BufferedWriter,
             new_val=None,
             endian=ENDIAN):
    assert fin is not None
    assert fout is not None

    val = int.from_bytes(fin.read(SIZE_U32), endian, signed=False)
    to_write = new_val if new_val is not None else val
    fout.write(to_write.to_bytes(SIZE_U32, endian, signed=False))

    return to_write

class GGMLTensor:
    """GGML model layer weights
    
    do we even need this class if we're just reading weights and re-writing them?
    """
    def __init__(self, n_dims, ftype, dimensions, name):
        self.n_dims = n_dims
        self.ftype = ftype
        self.dimensions = dimensions
        self.name = name

    @staticmethod
    def from_bin(fin: io.BufferedReader) -> 'GGMLTensor':
        # number of dimension in tensor shape
        n_dims = read_i32(fin) 
        # Length of part name string
        length = read_i32(fin)
        # float type. \in [0-3], st. 0 = f32, 1 = f16, 2 = q4_0, 3 = q4_1
        ftype = read_i32(fin)


        raise NotImplementedError("TODO: read dimensions")

# ==============================================================================
# ==============================================================================
# ==============================================================================

def convert_to_multipart(args):
    """Converts a single-part ggml file into several multipart ggml files

    When in multipart format, the `target`.bin file contains header information.
    This includes, in order,
    1. the magic constant
    2. model hyperparameters
    3. tokenizer vocabulary

    Then, following files, `target`.bin.{0..n} contain the model weights. Each
    file also starts with some frontmatter, which appears to be, in order::
    1. n_dims (int32_t) (tensor shape)
    2. length (int32_t) (length of the part name)
    3. ftype (int32_t) (0 = f32, 1 = f16, 2 = q4_0, 3 = q4_1)
    4. dimensions (int32_t * n_dims) (note: appears this must be less than 2?)
    5. part name (char * length)

    What happens next depends if split is by row (1) or column (0). This is
    specified by the part name, which corresponds to a layer in the model (or
    tokenizer embeddings). They follow this format:
    ```
    split_type = 0:
    regex:
      - tok_embeddings.*
      - layers.*.attention.wo.weight
      - layers.*.feed_forward.w2.weight

    split_type = 1:
    regex:
      - output.*
      - layers.*.attention.wq.weight
      - layers.*.attention.wk.weight
      - layers.*.attention.wv.weight
      - layers.*.feed_forward.w1.weight
      - layers.*.feed_forward.w3.weight
    ```
    
    This is specified by the part name, using this cpp code:
    ```cpp
    if (name.find("tok_embeddings") != std::string::npos) {
        split_type = 0;
    } else if (name.find("layers") != std::string::npos) {
        if (name.find("attention.wo.weight") != std::string::npos) {
            split_type = 0;
        } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
            split_type = 0;
        } else {
            split_type = 1;
        }
    } else if (name.find("output") != std::string::npos) {
        split_type = 1;
    }
    ```
    

    :param args: _description_
    :type args: _type_
    :raises FileNotFoundError: target model weight file does not exist
    :raises ValueError: _description_
    """

    # check that file exists
    filename = args["filename"]
    if not os.path.exists(filename):
        # raise FileNotFoundError(f"File {filename} does not exist")
        print(f"File {filename} does not exist")

    # Store the original file, just in case
    if not os.path.exists(f"{filename}.bak"):
        print(f"Backing up {filename} to {filename}.bak")
        shutil.copyfile(filename, f"{filename}.bak")

    # Remove original so we can use it to store frontmatter. We'll read from the
    # backup.
    os.remove(filename)

    print(f"Converting {filename}")

    # fin = open(filename, 'rb')
    with open(f"{filename}.bak", 'rb') as fin:
        with open(f"{filename}", 'wb+') as fout:
            # Read and validate the magic number
            magic = pipe_u32(fin, fout)
            if magic != EXPECTED_MAGIC:
                raise ValueError(f"File {filename} does not appear to be a ggml file, or is of a different version. Expected magic number {EXPECTED_MAGIC}, got {magic}")

            # Load in model hyperparameters
            # vocab, embd, mult, head, layer, rot, f16
            hparams = {}
            hparams["n_vocab"] = pipe_i32(fin, fout)
            hparams["n_embd"] = pipe_i32(fin, fout)
            hparams["n_mult"] = pipe_i32(fin, fout)
            hparams["n_head"] = pipe_i32(fin, fout)
            hparams["n_layer"] = pipe_i32(fin, fout)
            hparams["n_rot"] = pipe_i32(fin, fout)
            hparams["f16"] = pipe_i32(fin, fout)

            print(hparams)

            vobab = parse_vobab(fin, fout, n_vocab=hparams["n_vocab"])
            print(vobab)

def parse_vobab(fin: io.BufferedReader,
                fout: io.BufferedWriter,
                n_vocab: int):
    """read vocab
    each word in vovab is <len><word>, where len is a 32 bit signed integer

    :param fin: _description_
    :type fin: io.BufferedReader
    :param n_vocab: _description_
    :type n_vocab: int
    """
    vocab = {
        "word_to_idx": {},
        "idx_to_word": {},
    }
    for i in range(n_vocab - 1):
        l = pipe_u32(fin, fout)
        # print(f"{i}: {l}", end="\n")
        word_bin = fin.read(l)
        fout.write(word_bin)
        word = word_bin.decode(errors='replace')
        print(f"{i}: ({l}) {word}", end="\n")
        vocab["word_to_idx"][word] = i
        vocab["idx_to_word"][i] = word

    vocab["word_to_idx"]["<pad>"] = n_vocab - 1
    vocab["idx_to_word"][n_vocab - 1] = "<pad>"

    return vocab

        #if (i < 30000):
        #    print("{}: vocab[{}] = '{}'".format(__func__, i, word))
def decode_and_reencode_layer_as_ggml_seg(
    fin: io.BufferedReader,
    fout: io.BufferedWriter,
    hparams: dict,
):
    """encode a llama layer as a file
    """
    n_dims = read_i32(fin)
    name_length = read_i32(fin)
    ftype = read_i32(fin)
    # sanity check on ftype
    if ftype != hparams["f16"]:
        print(f"Warning: layer has ftype {ftype}, but model hyperparameter specify ftype {hparams['f16']}", file=sys.stderr)
    pass

if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.description = 'Convert a single-part ggml file into several multipart ggml files'
    # parser.add_argument('filename', help='filename to convert', default='gpt4all-lora-quantized.bin')
    # parser.add_argument('--num-parts', type=int, default=2, help='number of parts to split the file into (default: 2)')
    # args = parser.parse_args()
    args = {}
    args["filename"] = 'gpt4all-lora-quantized.bin'
    # args["num_parts"] = 
    
    # with open('gpt4all-lora-quantized.bin', 'rb') as f:
    #     data = f.read()
    convert_to_multipart(args)
