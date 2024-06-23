import pickle
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import torch
import numpy as np
import random
from .AsmVocab import AsmVocab
from time import time


class PackerBertDataset(Dataset):
    def __init__(self, db_path: str, vocab: AsmVocab, workers, mlm_probability=0.167):
        self.db_path = db_path
        self.vocab = vocab
        self.workers = workers
        self.mlm_probability = mlm_probability
        self.inputs, self.labels = self.encode()
        self.sample_len = len(self.inputs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        output = {"bert_input": self.inputs[index],  # 完整的(这里是包含two sentences)一条 bert sample, 包含各种标记tokens
                  "bert_label": self.labels[index]}  # bert_input在MLM Task情况下对应的bert_label, 注意此时不包含NSP Task的label

        return {key: torch.tensor(value) for key, value in output.items()}

    def encode_sample(self, sample_str):
        tokens = sample_str.split()
        tokens_idx = [self.vocab.word2id.get(token, self.vocab.unk_index) for token in tokens]
        inputs, labels = self.mask_tokens(tokens_idx)
        return inputs, labels

    def encode(self):
        with open(self.db_path, 'rb') as f:
            db_samples = pickle.load(f)
        merged_inputs = []
        merged_labels = []
        for db_sample in db_samples:
            inputs, labels = self.encode_sample(db_sample)
            merged_inputs.append(inputs)
            merged_labels.append(labels)
        return merged_inputs, merged_labels

    def mask_tokens(self, tokens_idx):
        eos_id = self.vocab.eos_index
        eos_count = tokens_idx.count(eos_id)
        adjusted_mlm_probability = self.mlm_probability * len(tokens_idx) / (len(tokens_idx) - eos_count)
        labels = np.zeros(512)

        opcode_mask_flag = False
        for i, token_idx in enumerate(tokens_idx):
            if tokens_idx[i] == eos_id:
                opcode_mask_flag = False
                continue
            prob = random.random()
            if prob < adjusted_mlm_probability:
                if tokens_idx[i - 1] == eos_id:  # 第一个判断
                    opcode_mask_flag = True
                else:
                    if opcode_mask_flag:
                        continue
                prob = random.random()
                if prob < 0.8:
                    tokens_idx[i] = self.vocab.mask_index
                elif prob < 0.9:
                    tokens_idx[i] = random.randrange(len(self.vocab))
                labels[i] = token_idx

        inputs = np.pad(tokens_idx, (0, 512 - len(tokens_idx)), mode='constant', constant_values=0)
        return inputs, labels
        # nsp

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]


# 手动测试, 不存在于原始代码中
if __name__ == '__main__':
    start_time = time.time()
    vocab = AsmVocab()
    vocab.load("vocab.txt")
    train_dataset = PackerBertDataset(r'D:/dataset_0.pkl', vocab, workers=8)
    end_time = time.time()  # 记录结束时间
    print(end_time - start_time)
    train_data_loader = DataLoader(train_dataset, 8, num_workers=4)

    data_iter = tqdm(enumerate(train_data_loader),
                     desc="EP_%s:%d" % ('train', 1),
                     total=len(train_data_loader),
                     bar_format="{l_bar}{r_bar}")

    for i, data in data_iter:
        # 0. batch_data will be sent into the device(GPU or cpu)
        data = {key: value.to('cuda') for key, value in data.items()}
        print(data['bert_input'].shape, data['bert_input'].dtype, data['bert_label'].shape, data['bert_label'].dtype)
