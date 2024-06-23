import os
import pickle
import random
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import transformers
from loguru import logger
from tqdm import tqdm
from copy import deepcopy
from PackerBert.model.classify import CLASSIFIER
from PackerBert.dataset.AsmVocab import AsmVocab
from PackerBert.dataset.dataset_classify import PackerBertDataset
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix


def set_seed(seed):
    """
     Same seed makes sure The Experiment can be Reproduced
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ClassifyTrainer:

    def __init__(self, model: CLASSIFIER, vocab: AsmVocab, current_model_save_path: str, best_model_save_path: str,
                 train_path: str, val_path: str, num_workers: int, max_len: int, batch_size: int, train_size: int,
                 visualizer: SummaryWriter, epochs: int, lr: float = 1e-4, betas=(0.9, 0.999),
                 weight_decay: float = 0.01, warmup_ratio=0.1,
                 with_cuda: bool = True, cuda_devices=None, log_freq=10, random_seed=66):

        set_seed(random_seed)
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:1" if cuda_condition else "cpu")

        self.model = model.to(self.device)  # key component
        self.vocab = vocab  # key component

        logger.info(f'Total Bert DCC Parameters:{sum([p.nelement() for p in self.model.parameters()])}')

        # logger.error(
        #     f'Training from '
        #     f'epoch:{self.current_epoch}, '
        #     f'idx_in_dataloader:{self.current_dataloader_idx},'
        #     f'checkpoint_best_score:{self.checkpoint_best_score}')

        # 基础配置信息读取

        self.current_model_save_path = current_model_save_path
        self.best_model_save_path = best_model_save_path
        self.train_path = train_path
        self.val_path = val_path
        self.num_workers = num_workers
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_size = train_size  # 计算总训练步数
        self.total_train_steps = -(-self.train_size // batch_size) * self.epochs
        self.visualizer = visualizer

        # read from checkpoint
        self.current_epoch = 0
        self.current_dataloader_idx = 0
        self.checkpoint_best_score = 9999.0
        self.all_predictions = []
        self.all_labels = []
        self.sum_loss = 0

        # if with_cuda and torch.cuda.device_count() > 1:
        #     logger.info("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=[2])
        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      warmup_ratio * self.total_train_steps,
                                                                      self.total_train_steps)

        self.criterion = nn.CrossEntropyLoss()
        self.log_freq = log_freq

    def classify_train(self):
        for current_epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = current_epoch
            self.train()
            self.validate()

    def iteration(self, data_loader, train=True):  # TP, loss这些使用参数传入吧
        code_str = "train" if train else "validate"
        data_loader_iter = iter(data_loader)
        for _ in range(self.current_dataloader_idx):
            next(data_loader_iter)

        data_iter = tqdm(enumerate(data_loader_iter, start=self.current_dataloader_idx),
                         initial=self.current_dataloader_idx,
                         desc="%s_epoch_%d" % (code_str, self.current_epoch),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")

        for i, batch in data_iter:
            data = {key: value.to(self.device) for key, value in batch.items()}
            inputs = data["input"].to(self.device)
            labels = data["label"].to(self.device)

            dcc_output = self.model(inputs, (inputs == 0).sum(dim=1))
            predictions = torch.argmax(dcc_output, dim=-1)
            loss = self.criterion(dcc_output, labels)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if (i + 1) % 100 == 0:
                    self.current_dataloader_idx = i + 1
                    self.save(self.current_model_save_path)
            self.sum_loss += loss.item()
            self.all_predictions.extend(predictions.cpu().numpy())
            self.all_labels.extend(labels.cpu().numpy())

        if train:
            self.current_dataloader_idx = len(data_loader) - 1
            self.save(self.current_model_save_path)
            self.save(f'/home/lishijia/DeepPacker/PackerLM/checkpoints/task1/ep{self.current_epoch}.pt')

        mean_loss = self.sum_loss / len(data_loader)
        f1 = f1_score(self.all_labels, self.all_predictions)
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        precision = precision_score(self.all_labels, self.all_predictions)
        recall = recall_score(self.all_labels, self.all_predictions)
        logger.info(
            f'Current epoch{self.current_epoch} {code_str} loss:{mean_loss} f1:{f1} accuracy {accuracy} recall {recall} precision {precision} ')

        self.all_predictions = []
        self.all_labels = []
        self.sum_loss = 0
        self.current_dataloader_idx = 0
        return mean_loss, f1, accuracy, recall, precision

    def train(self):
        self.model.train()
        train_path = self.train_path
        train_dataset = ClassifyDataset(train_path, self.vocab, self.max_len)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=0)
        loss, f1, accuracy, recall, precision = self.iteration(train_dataloader, train=True)
        self.visualizer.add_scalars('Train/task1',
                                    {'loss': loss, 'f1': f1, "accuracy": accuracy, "recall": recall,
                                     "precision": precision},
                                    self.current_epoch)

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            val_path = self.val_path
            val_dataset = ClassifyDataset(val_path, self.vocab, self.max_len)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0)
            loss, f1, accuracy, recall, precision = self.iteration(val_dataloader, train=False)
            self.visualizer.add_scalars('Validation/task1',
                                        {'loss': loss, 'f1': f1, "accuracy": accuracy, "recall": recall,
                                         "precision": precision},
                                        self.current_epoch)

            if loss < self.checkpoint_best_score:
                logger.info(f'Current epoch{self.current_epoch} validation loss:{loss} is better than '
                            f'last epoch model loss:{self.checkpoint_best_score}. '
                            f'Save this model as checkpoint_best.pt. ')
                self.checkpoint_best_score = loss
                self.save(self.best_model_save_path)

    def save(self, save_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'current_dataloader_idx': self.current_dataloader_idx,
            'sum_loss': self.sum_loss,
            "all_predictions": self.all_predictions,
            "all_labels": self.all_labels,
            'checkpoint_best_score': self.checkpoint_best_score,
        }, save_path)

