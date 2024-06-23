import os
import pickle
import random
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
from PackerBert.model.pretrain_tasks import BERTLM
from PackerBert.trainer.optim_schedule import ScheduledOptim
from copy import deepcopy
from PackerBert.dataset.AsmVocab import AsmVocab
from PackerBert.dataset.dataset import PackerBertDataset
from glob import glob
from time import time


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
    # # some cudnn methods can be random even after fixing the seed
    # # unless you tell it to be deterministic
    # torch.backends.cudnn.deterministic = True


class BERTTrainer:

    def __init__(self, model: BERTLM, vocab: AsmVocab, current_model_save_path: str,
                 best_model_save_path: str, train_path: str, val_path: str, num_workers: int,
                 max_len: int, batch_size: int, train_size: int, visualizer: SummaryWriter,
                 epochs: int, lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_ratio=0.01,
                 with_cuda: bool = True, cuda_devices=None, log_freq=10, random_seed=66):

        set_seed(random_seed)
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.model = model.to(self.device)  # key component
        self.vocab = vocab

        logger.info(f'Total Bert Language Model Parameters:{sum([p.nelement() for p in self.model.parameters()])}')

        self.current_epoch = 0
        self.current_dataset = 0
        self.current_checkpoint_best_score = 9999.0

        self.current_model_save_path = current_model_save_path
        self.best_model_save_path = best_model_save_path
        self.train_path = train_path
        self.val_path = val_path
        self.num_workers = num_workers
        self.max_len = max_len
        self.batch_size = batch_size
        self.train_size = train_size
        self.visualizer = visualizer
        self.epochs = epochs
        self.lr = lr

        self.total_train_steps = self.epochs * (-(-train_size // batch_size))

        if with_cuda and torch.cuda.device_count() > 1:
            logger.info("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      warmup_ratio * self.total_train_steps,
                                                                      self.total_train_steps)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.log_freq = log_freq
        self.avg_loss = []
        self.avg_ppl = []
        self.ppl_base = 2

    def pretrain(self):
        for current_epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = current_epoch
            print(f"---------epoch{self.current_epoch} start--------")
            self.train()
            torch.save(self.model.bert.state_dict(), f"E:/DeepPacker/PackerLM/checkpoints/ep{self.current_epoch}.pt")
            # self.save(f"E:/DeepPacker/PackerLM/checkpoints/ep{self.current_epoch}.pt")
            # self.save(self.current_model_save_path)
            # self.validate()
            self.current_dataset = 0

    def iteration(self, data_loader, data_index, train=True):
        code_str = "train" if train else "validate"
        sum_loss = 0.0

        data_iter = tqdm(enumerate(data_loader),
                         desc="%s_dataset_%d" % (code_str, data_index),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")

        for i, batch in data_iter:
            data = {key: value.to(self.device) for key, value in batch.items()}
            mlm_inputs = data["bert_input"]
            mlm_label = data['bert_label'].long()
            mask_lm_output = self.model(mlm_inputs)
            loss = self.criterion(mask_lm_output.transpose(1, 2), mlm_label)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            sum_loss += loss.item()
        mean_loss = round(sum_loss / len(data_loader), 3)
        mean_ppl = round(self.ppl_base ** mean_loss, 3)

        post_fix = {
            "epoch": self.current_epoch,
            "dataset": data_index,
            "MLM_loss:": mean_loss,
            "MLM_ppl:": mean_ppl,
            # "lr" : self.optimizer.param_groups[0]["lr"]
            "lr:": self.scheduler.get_last_lr()
        }
        data_iter.write(str(post_fix))
        return mean_loss, mean_ppl

    def train(self):

        self.model.train()
        dataset_paths = glob('{}/*.pkl'.format(self.train_path))

        for dataset_i in range(self.current_dataset, len(dataset_paths)):
            dataset_path = dataset_paths[dataset_i]
            train_dataset = PackerBertDataset(dataset_path, self.vocab, workers=self.num_workers)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=0)
            loss, ppl = self.iteration(train_dataloader, data_index=dataset_i, train=True)
            self.visualizer.add_scalars('Train/MLM',
                                        {'loss': loss, 'ppl': ppl},
                                        (self.current_epoch) * len(dataset_paths) + dataset_i)
            if (dataset_i + 1) % 2 == 0:
                self.current_dataset = dataset_i + 1
                self.save(self.current_model_save_path)
            if (dataset_i + 1) % 20 == 0:
                self.save(f"E:/DeepPacker/PackerLM/checkpoints/ep{self.current_epoch}_data{dataset_i}.pt")

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            sum_loss = 0.0
            dataset_paths = glob('{}/*.pkl'.format(self.val_path))
            for dataset_i in range(len(dataset_paths)):
                validate_dataset = PackerBertDataset(dataset_paths[dataset_i], self.vocab, workers=self.num_workers)
                validate_dataloader = DataLoader(validate_dataset, batch_size=self.batch_size, num_workers=0)
                loss, ppl = self.iteration(validate_dataloader, data_index=dataset_i, train=False)
                sum_loss += loss

            mean_loss = round(sum_loss / len(dataset_paths), 3)

            logger.info(f'Average validation loss in epoch{self.current_epoch} is: {mean_loss}')
            if mean_loss < self.current_checkpoint_best_score:
                logger.info(f'Current epoch{self.current_epoch} validation loss:{mean_loss} is better than '
                            f'last epoch model loss:{self.current_checkpoint_best_score}. '
                            f'Save this model as checkpoint_best.pt. ')
                self.current_checkpoint_best_score = mean_loss
                torch.save(self.model.bert.state_dict(), self.best_model_save_path)

            self.visualizer.add_scalars('Validation/MLM',
                                        {'loss': mean_loss,
                                         'ppl': round(self.ppl_base ** mean_loss, 3)},
                                        self.current_epoch)

    def save(self, save_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'current_dataset': self.current_dataset,
            'current_checkpoint_best_score': self.current_checkpoint_best_score
        }, save_path)
