from config import *
import shutil
import os
import torch
from config_classify import configs
from PackerBert.model.bert_framework import BERT
from PackerBert.model.classify import CLASSIFIER
from PackerBert.trainer.train_classify import ClassifyTrainer
from PackerBert.dataset.AsmVocab import AsmVocab
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

class UsableTransformer:
    def __init__(self):
        vocab_path = configs["vocab_path"]
        vocab_size = configs["vocab_size"]
        pretrain_model_save_path = configs["pretrain_model_save_path"]

        # bert config
        hidden = configs["hidden"]
        n_layers = configs["n_layers"]
        attn_heads = configs["attn_heads"]

        num_workers = configs["num_workers"]
        self.max_len = configs["max_len"]
        self.device = torch.device("cuda:0")
        batch_size = configs["batch_size"]
        epochs = configs["epochs"]
        lr = configs["lr"]
        betas = configs["betas"]
        weight_decay = configs["weight_decay"]
        warmup_ratio = configs["warmup_ratio"]
        with_cuda = configs["with_cuda"]
        cuda_devices = configs["cuda_devices"]
        log_freq = configs["log_freq"]

        self.vocab = AsmVocab()
        self.vocab.load(vocab_path)

        logger.info(f"Bert config: hidden:{hidden}, n_layers:{n_layers}, attn_heads:{attn_heads}")
        logger.info(f"Dataset config:  max_len:{self.max_len}, batch_size:{batch_size} ")
        logger.info(f"Train config: epochs:{epochs}, initial lr:{lr}, betas:{betas}, weight_decay:{weight_decay}, "
                    f"warmup_ratio:{warmup_ratio}, with_cuda:{with_cuda}, cuda_devices:{cuda_devices},")
        logger.info(f"Log config: log_freq:{log_freq}")
        logger.info(f"Vocab size:{vocab_size}")

        torch.cuda.set_device(0)

        self.bert = BERT(vocab_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads).to("cuda:0")

        checkpoint = torch.load(pretrain_model_save_path, map_location="cuda:0")
        self.bert.load_state_dict(checkpoint)

    def encode(self, text):
        tokens = text.split()
        tokens_idx = [self.vocab.word2id.get(token, self.vocab.unk_index) for token in tokens]
        tokens_idx.extend([0] * (self.max_len - len(tokens_idx)))
        tokens_idx=torch.LongTensor(tokens_idx)
        tokens_idx=tokens_idx.to(self.device)
        tokens_idx=tokens_idx.unsqueeze(0)
        result=self.bert.forward(tokens_idx)
        return result.to('cpu')





