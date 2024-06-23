import os
import shutil
import time

import torch
from PackerBert.model.bert_framework import BERT
from PackerBert.dataset.vocab import WordVocab
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    if os.path.exists('../visualizer/TokenSpace'):
        shutil.rmtree('../visualizer/TokenSpace')
    visualizer = SummaryWriter('../visualizer/TokenSpace')

    vocab = WordVocab.load_vocab(r'dataset/corpus.test')
    vocab_size = len(vocab)

    # make sure the hyperparameters are strictly the same
    pm = BERT(vocab_size, hidden=768, n_layers=12, attn_heads=12).to('cuda:0')

    # load the pretrained bert model to get token embedding
    pm.load_state_dict(torch.load(r'../checkpoints/bert.model.ep9').state_dict())  # ?这个方法还和文件夹位置相关?!
    tokens = torch.arange(5, vocab_size).long().cuda()  # all valid tokens

    embeddings = pm.embedding.token(tokens)

    labels = vocab.from_seq(tokens)

    visualizer.add_embedding(embeddings, labels)
    print("Here!")
    time.sleep(10)
