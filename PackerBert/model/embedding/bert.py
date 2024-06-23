import torch
import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are checkpoints of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        # 下面是参考hugging face的position embedding过程, 略有不同, 原始代码请参考其源程序
        # 此时声明的position embedding应该已经更改为learnable
        padding_idx = 0  # 在此方案的vocabulary中, <pad> = 0
        bit_mask = sequence.ne(padding_idx).int()
        # input tokens(index)中不等于<pad>的位置为1, 反之为0，通过累加和求, 求出非<pad> tokens的位置坐标, 最后的'*mask'的作用是:
        # 保留非<pad> token的位置坐标, <pad> token的位置坐标全为0，这种情况正好符合position embedding中的padding_idx=0
        position_index = torch.cumsum(bit_mask, dim=1).type_as(bit_mask) * bit_mask
        x = self.token(sequence) + self.position(position_index.long())

        # transformer说不仅对embedding sum进行dropout, 还要对position encoding进行dropout, 但是bert用的是embedding, 所以不再dropout了
        return self.dropout(x)
