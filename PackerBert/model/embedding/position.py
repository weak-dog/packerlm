import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Embedding):
    def __init__(self, d_model, max_len=512+1):  # 加上1是因为在求非<pad> tokens的位置时, 由于cumsum的特性, 那么有效起始位置的必定为1
        super().__init__(max_len, d_model, padding_idx=0)  # 仍然需要padding_idx=0, 因为position里还存在0
