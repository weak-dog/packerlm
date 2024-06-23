import os

import torch
import torch.nn as nn
from config_classify import configs
from PackerBert.model.bert_framework import BERT


class CLASSIFIER(nn.Module):

    def __init__(self, bert: BERT):
        super().__init__()

        self.bert = bert
        self.downstream_dcc = Classification(self.bert.hidden, self.bert.hidden, configs["num_classes"])

    def forward(self, x, pad_len):
        x = self.bert(x)
        return self.downstream_dcc(x, pad_len)


class Classification(nn.Module):
    def __init__(self, hidden, inner_dim, num_classes):
        super().__init__()
        self.dense = nn.Linear(hidden, inner_dim)
        self.activation_fn = nn.ReLU()
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x, pad_len):
        x = torch.stack([torch.mean(k[:len(k) - m, :], dim=0) for k, m in zip(x, pad_len)])
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.out_proj(x)

        return x
