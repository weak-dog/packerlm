import torch.nn as nn

from PackerBert.model.encoder import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        # self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)

        self.attention = nn.MultiheadAttention(embed_dim=hidden, num_heads=attn_heads,
                                               dropout=dropout, batch_first=True)

        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))

        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x,  # q,k,v
                                                             key_padding_mask=mask, need_weights=False))

        x = self.output_sublayer(x, self.feed_forward)  # 理论上此句应该如下句，即格式和上一句相同，但实际效果都一样
        # x = self.output_sublayer(x, lambda _x: self.feed_forward.forward(_x))

        # 这里不应该再次dropout了, transformer的原文是这样描述的：
        # We apply dropout[27] to the checkpoints of each sub-layer, before it is added to the
        # sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
        # positional encodings in both the encoder and decoder stacks.
        # 所以也没说每个数据输入encoder stack前都要进行dropout啊?难道确实需要dropout?因为上面说的是stacks, 而不是stack
        # 意思是难道是：作为完整的迭代过程, 因为embedding sum是输入第一个stack的数据, 它进行了dropout, 那其他stack也应该dropout？
        # 但是他这个书面上确实只针对embedding. 根据后续实验结果再决定是否dropout吧, 目前先不dropout了。
        # return self.dropout(x)  # torch的官方实现也没有dropout
        return x
