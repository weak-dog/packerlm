'''
tokenize & save tokenized insns
build vocab
'''
import os
import re
import pickle
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from AsmVocab import AsmVocab
import json
from collections import Counter

# 统计词频
class VocabGenerator:
    def __init__(self, norm_dir, vocab_dir, vocab_path, workers=4):
        self.norm_dir = norm_dir
        self.vocab_dir = vocab_dir
        self.vocab_path = vocab_path
        self.workers = workers
        os.makedirs(self.vocab_dir, exist_ok=True)

    def gen_vocab_perfile(self, json_file):
        output_json = os.path.join(self.vocab_dir, os.path.basename(json_file))
        with open(json_file) as f:
            norm_res = json.load(f)
        token_vocab = {}
        for section in norm_res:
            regions=norm_res[section]
            for region in regions:
                ins_list=region[1]
                for ins in ins_list:
                    ins = ins.split(";")[3]
                    split_ins=ins.split("\t")
                    ins_tokens=[]
                    opcode = "_".join(split_ins[0].split())
                    ins_tokens.append(opcode.strip())
                    if len(split_ins) > 1:
                        for operand in split_ins[1].split(","):
                            operand = operand.strip()
                            if operand:
                                ins_tokens.append(operand)
                    for token in ins_tokens:
                        if token not in token_vocab:
                            token_vocab[token]=1
                        else:
                            token_vocab[token] += 1
        with open(output_json, 'w') as f:
            f.write(json.dumps(token_vocab))

    def merge_vocab(self):
        vocab = {}
        for vocab_file in tqdm(glob('{}/*.json'.format(self.vocab_dir))):
            with open(vocab_file, 'r') as f:
                vocab_perfile=json.load(f)
                vocab = dict(Counter(vocab) + Counter(vocab_perfile))
        sorted_vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        sorted_vocab_dict = {k: vocab[k] for k in sorted_vocab_list}
        with open("vocab.json","w")as f:
            f.write(json.dumps(sorted_vocab_dict, indent=4))
        print(len(sorted_vocab_list))
        vocab = AsmVocab(sorted_vocab_list)
        vocab.save(self.vocab_path)

    def gen_vocab(self):
        json_files=[]
        for json_file in glob('{}/*.json'.format(self.norm_dir)):
            dest_path= os.path.join(self.vocab_dir, os.path.basename(json_file))
            if not os.path.exists(dest_path):
                json_files.append(json_file)
        with ProcessPoolExecutor(max_workers=self.workers) as executor, tqdm(total=len(json_files)) as progress_bar:
            for _ in executor.map(self.gen_vocab_perfile, json_files, ):
                progress_bar.update(1)
        self.merge_vocab()

if __name__ == '__main__':
    with open("configs/Config.json")as f:
        configs=json.load(f)
    if configs["gen_vocab"]:
        vocab_generator=VocabGenerator(configs["norm_dir"], configs["vocab_dir"], configs["vocab_path"], configs["vocab_threads"])
        vocab_generator.gen_vocab()
    

