import pickle

BASE_DIR = "/home/sujayrokade/hsynthesis"
with open(f"{BASE_DIR}/src/ctoi.txt", "rb") as file:
    enc_dict = pickle.load(file)

SRC_DIR = f"{BASE_DIR}/IAM"
OUT_DIR = f"{BASE_DIR}/src/out"
RUNS_DIR = f"{BASE_DIR}/src/runs"
BATCH_SIZE = 64
NUM_TOKENS = len(enc_dict)
EMBEDDING_SIZE = 128
NUM_LAYERS = 4  # Ideally 4
PADDING_IDX = 0
Z_LEN = 128  # Z_LEN should be equal to EMBEDDING_SIZE?
CHUNKS = 8
CBN_MLP_DIM = 512
RELEVANCE_FACTOR = 1
LEARNING_RATE = 2e-4
BETAS = (0, 0.999)
