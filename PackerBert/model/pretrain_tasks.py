import os

import torch
import torch.nn as nn
from config import configs
from .bert_framework import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()

        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x):
        x = self.bert(x)
        return self.mask_lm(x)

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: checkpoints size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # return self.softmax(self.linear(x))  # 把LogSoftmax放到了pretrain中
        return self.linear(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model checkpoints size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        # self.softmax = nn.LogSoftmax(dim=-1)  # 把LogSoftmax放到了pretrain中

    def forward(self, x):
        # return self.softmax(self.linear(x[:, 0]))
        return self.linear(x[:, 0])
