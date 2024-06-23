import torch.nn as nn

from .encoder.transformer import TransformerBlock

from .embedding import BERTEmbedding


class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()

        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # For a binary mask, a True value indicates that
        # the corresponding key value will be ignored for the purpose of attention.
        mask = (x == 0)  # <pad>

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def encode(self, x, encode_layers: int):
        """编码最终要经过多少层transformer encoders"""
        assert encode_layers > 0
        mask = (x == 0)
        x = self.embedding(x)
        for transformer in self.transformer_blocks[:encode_layers]:
            x = transformer.forward(x, mask)
        return x

