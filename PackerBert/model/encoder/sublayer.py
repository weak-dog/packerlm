import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()

        # self.norm = LayerNorm(size)
        # 使用torch官方实现的LayerNorm, 默认eps=1e-5
        self.norm = nn.LayerNorm(size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # return x + self.dropout(sublayer(self.norm(x)))

        # 按照原本的transformer encoder,LayerNorm应该在sublayer的输出dropout和residual add之后进行
        return self.norm(x + self.dropout(sublayer(x)[0]))  # 这里的sublayer是lambda函数，指的是MultiHeadedAttention的forward
