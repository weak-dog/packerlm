import re
import pickle
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import os
from tqdm import tqdm
from glob import glob
import json
import threading
import random
import shutil
from collections import Counter
import random

palmtree_remove_prefix = ["ptr", "short", "offset"]
palmtree_replace_prefix = {
    "qword": "word",
    "tword": "word",
    "zword": "word",
    "yword": "word",
    "oword": "word",
    "tbyte": "byte"
}


class GenClassifyDataset():
    def __init__(self, data_num, label, chunk_size, sample_num, norm_dir, dataset_dir, drop_threshold=0.8,
                 split_region=True):
        self.data_num = data_num  # 数据集大小
        self.label = label  # 0-data, 1-code, 2-packed-data
        self.chunk_size = chunk_size  # 一个chunk中的指令数
        self.sample_num = sample_num  # 从一个程序中抽样多少组数据
        self.norm_dir = norm_dir
        self.root_dir = dataset_dir
        self.drop_threshold = drop_threshold
        self.split_region = split_region
        self.finished_num = 0
        self.dataset = []  # [deeppacker, palmtree, xda, label]
        self.dataset_palmtree = []
        self.dataset_deeppacker = []
        self.dataset_xda = []

    @staticmethod
    def parse_instruction_palmtree(ins):
        for prefix in palmtree_remove_prefix:
            ins = ins.replace(prefix, "")
        for prefix in palmtree_replace_prefix:
            ins = ins.replace(prefix, palmtree_replace_prefix[prefix])
        ins = re.sub('\s+', ', ', ins, 1)
        ins = re.sub("\S+\[", "[", ins)
        parts = ins.split(', ')
        operand = []
        token_lst = []
        if len(parts) > 1:
            operand = parts[1:]
        token_lst.append(parts[0])
        for i in range(len(operand)):
            symbols = re.split('([0-9A-Za-z]+)', operand[i])
            symbols = [s.strip() for s in symbols if s]
            processed = []
            for j in range(len(symbols)):
                if symbols[j][:2] == '0x' and len(symbols[j]) > 6 and len(symbols[j]) < 15:
                    processed.append("address")
                else:
                    processed.append(symbols[j])
                processed = [p for p in processed if p]
            token_lst.extend(processed)
        return ' '.join(token_lst)

    @staticmethod
    def split_block_palmtree(block):
        ins_list = []
        for ins in block:
            if "data;" in ins:
                ins_list.append("data")
            else:
                ins = ins.split(";")[1]
                parsed_ins = GenClassifyDataset.parse_instruction_palmtree(ins)
                if parsed_ins:
                    ins_list.append(parsed_ins)
        ins_str = ','.join(ins_list)
        return ins_str, len(ins_list)

    @staticmethod
    def split_block_deeppacker(block):
        ins_list = []
        for ins in block:
            ins = ins.split(";")[3]
            split_ins = ins.split("\t")
            opcode = "_".join(split_ins[0].split())  # 处理两个opcode情况
            ins_list.append(opcode)
            if len(split_ins) > 1:
                for operand in split_ins[1].split(","):
                    operand = operand.strip()
                    if operand:
                        ins_list.append(operand)
            ins_list.append("<eos>")
        ins_str = " ".join(ins_list)
        return ins_str

    @staticmethod
    def split_block_xda(block, LE=False):
        ins_list = []
        keep = True
        for ins in block:
            if "data;" in ins:
                if "pad_normal" in ins:
                    ins_bytes = ins.split(";")[2]
                    if "00" in ins_bytes:
                        ins_list.append("00")
                    else:
                        ins_list.append("ff")
                else:
                    ins_bytes = ins.split(";")[2]
                    ins_bytes_split = [ins_bytes[i:i + 2] for i in range(0, len(ins_bytes), 2)]
                    ins_list.extend(ins_bytes_split)
            else:
                ins_bytes = ins.split(";")[2]
                ins_list.extend(ins_bytes.split())
        ins_str = ' '.join(ins_list)
        if LE:
            bytes_counts = Counter(ins_list)
            max_bytes, max_frequency = bytes_counts.most_common(1)[0]
            if max_frequency > 0.4 * len(ins_list):
                keep = False
        return ins_str, keep

    def gen_dataset(self, datatype):
        current_index = 0
        num = self.data_num / 10000
        norm_files = glob('{}/*.json'.format(self.norm_dir))
        random.shuffle(norm_files)
        with tqdm(total=self.data_num) as progress_bar:
            stop = False
            for json_file in norm_files:
                if stop:
                    break
                try:
                    with open(json_file) as f:
                        current_index += 1
                        if current_index % 100 == 0:
                            print(current_index)
                        dis_res = json.load(f)
                        for section in dis_res:
                            if stop:
                                break
                            section = dis_res[section]
                            if self.split_region:
                                regions = section
                                for region in regions:
                                    if stop:
                                        break
                                    if region[0] == "data":
                                        if datatype == "packed":
                                            region_type = 2
                                        else:
                                            region_type = 0
                                    else:
                                        region_type = 1
                                    if region_type != self.label:
                                        continue
                                    region_len = len(region[1])
                                    chunk_num = region_len // self.chunk_size
                                    if chunk_num:
                                        sample_num = min(self.sample_num, chunk_num)
                                        sampled_indexes = random.sample(range(0, chunk_num), sample_num)
                                        for sampled_index in sampled_indexes:
                                            if stop:
                                                break
                                            start_index = sampled_index * self.chunk_size
                                            chunk = region[1][start_index:start_index + self.chunk_size]
                                            block_ins_deeppacker = GenClassifyDataset.split_block_deeppacker(chunk)
                                            block_ins_palmtree, block_len_palmtree = GenClassifyDataset.split_block_palmtree(
                                                chunk)
                                            block_ins_xda, keep = GenClassifyDataset.split_block_xda(chunk,
                                                                                                     not self.split_region)
                                            if block_len_palmtree > self.chunk_size * self.drop_threshold:
                                                if [block_ins_deeppacker, self.label] not in self.dataset_deeppacker:
                                                    self.dataset.append(
                                                        [block_ins_deeppacker, block_ins_palmtree, block_ins_xda,
                                                         self.label])
                                                    self.finished_num += 1
                                                    progress_bar.update(1)
                                            if self.finished_num == self.data_num:
                                                stop = True
                            else:  # low entropy
                                section_len = len(section)
                                chunk_num = section_len // self.chunk_size
                                if chunk_num:
                                    sample_num = min(self.sample_num, chunk_num)
                                    sampled_indexes = random.sample(range(0, chunk_num), sample_num)
                                    for sampled_index in sampled_indexes:
                                        if stop:
                                            break
                                        start_index = sampled_index * self.chunk_size
                                        chunk = section[start_index:start_index + self.chunk_size]
                                        block_ins_deeppacker = GenClassifyDataset.split_block_deeppacker(chunk)
                                        block_ins_palmtree, block_len_palmtree = GenClassifyDataset.split_block_palmtree(
                                            chunk)
                                        block_ins_xda, keep = GenClassifyDataset.split_block_xda(chunk,
                                                                                                 not self.split_region)
                                        if block_len_palmtree > self.chunk_size * self.drop_threshold and keep:
                                            if [block_ins_deeppacker, self.label] not in self.dataset_deeppacker:
                                                self.dataset.append(
                                                    [block_ins_deeppacker, block_ins_palmtree, block_ins_xda,
                                                     self.label])
                                                self.finished_num += 1
                                                progress_bar.update(1)
                                        if self.finished_num == self.data_num:
                                            stop = True
                except:
                    continue

        # shuffle samples
        random.shuffle(self.dataset)
        for sample in self.dataset:
            self.dataset_deeppacker.append([sample[0], sample[3]])
            self.dataset_palmtree.append([sample[1], sample[3]])
            self.dataset_xda.append([sample[2], sample[3]])

        for model in ["palmtree", "xda", "deeppacker"]:
            dataset_dir = os.path.join(self.root_dir, model, f"{num}w")
            os.makedirs(dataset_dir, exist_ok=True)
            print(os.path.join(dataset_dir, f'{datatype}_{num}w.pkl'))
            with open(os.path.join(dataset_dir, f'{datatype}_{num}w.pkl'), "wb") as f:
                if model == "palmtree":
                    pickle.dump(self.dataset_palmtree, f)
                elif model == "xda":
                    pickle.dump(self.dataset_xda, f)
                elif model == 'deeppacker':
                    pickle.dump(self.dataset_deeppacker, f)
            with open(os.path.join(dataset_dir, f'{datatype}_{num}w.json'), 'w') as f:
                if model == "palmtree":
                    json.dump(self.dataset_palmtree, f)
                elif model == "xda":
                    json.dump(self.dataset_xda, f)
                elif model == 'deeppacker':
                    json.dump(self.dataset_deeppacker, f)

    # palmtree和deeppacker写入pkl，xda写入.data文件
    def merge_datasets(self, dataset_dir, out_dir):
        merged_train_dataset = []
        merged_valid_dataset = []
        merged_test_dataset = []
        for pkl_file in (glob('{}/*.pkl'.format(dataset_dir))):
            with open(pkl_file, 'rb') as f:
                dataset = pickle.load(f)
                merged_train_dataset.extend(dataset[:int(len(dataset) * 0.8)])
                merged_valid_dataset.extend(dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)])
                merged_test_dataset.extend(dataset[int(len(dataset) * 0.9):])
        random.shuffle(merged_train_dataset)
        random.shuffle(merged_valid_dataset)
        random.shuffle(merged_test_dataset)
        with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
            pickle.dump(merged_train_dataset, f)
        with open(os.path.join(out_dir, "val.pkl"), "wb") as f:
            pickle.dump(merged_valid_dataset, f)
        with open(os.path.join(out_dir, "test.pkl"), "wb") as f:
            pickle.dump(merged_test_dataset, f)

    def merge_datasets_xda(self, dataset_dir, out_dir):
        merged_train_dataset = []
        merged_valid_dataset = []
        merged_test_dataset = []
        for pkl_file in (glob('{}/*.pkl'.format(dataset_dir))):
            with open(pkl_file, 'rb') as f:
                dataset = pickle.load(f)
                train_dataset = dataset[:int(len(dataset) * 0.8)]
                valid_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
                test_dataset = dataset[int(len(dataset) * 0.9):]
                merged_train_dataset.extend(train_dataset)
                merged_valid_dataset.extend(valid_dataset)
                merged_test_dataset.extend(test_dataset)
        random.shuffle(merged_train_dataset)
        random.shuffle(merged_valid_dataset)
        random.shuffle(merged_test_dataset)
        with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
            pickle.dump(merged_train_dataset, f)
        with open(os.path.join(out_dir, "val.pkl"), "wb") as f:
            pickle.dump(merged_valid_dataset, f)
        with open(os.path.join(out_dir, "test.pkl"), "wb") as f:
            pickle.dump(merged_test_dataset, f)
        with open(os.path.join(out_dir, "train.data"), "w") as f:
            for data in merged_train_dataset:
                f.write(data[0] + "\n")
        with open(os.path.join(out_dir, "valid.data"), "w") as f:
            for data in merged_valid_dataset:
                f.write(data[0] + "\n")
        with open(os.path.join(out_dir, "test.data"), "w") as f:
            for data in merged_test_dataset:
                f.write(data[0] + "\n")
        with open(os.path.join(out_dir, "train.label"), "w") as f:
            for data in merged_train_dataset:
                f.write(str(data[1]) + "\n")
        with open(os.path.join(out_dir, "valid.label"), "w") as f:
            for data in merged_valid_dataset:
                f.write(str(data[1]) + "\n")
        with open(os.path.join(out_dir, "test.label"), "w") as f:
            for data in merged_test_dataset:
                f.write(str(data[1]) + "\n")


class GenPreTrainDataset():
    def __init__(self, chunk_size, norm_dir, pretrain_dir, split_region=True, workers=8):
        self.chunk_size = chunk_size  # 一个chunk中的指令数
        self.norm_dir = norm_dir
        self.pretrain_dir = pretrain_dir
        self.split_region = split_region
        self.workers = workers
        os.makedirs(self.pretrain_dir, exist_ok=True)

    def genpkl_perfile(self, norm_file):  # 排除指令阶段
        out_file = os.path.join(self.pretrain_dir, os.path.basename(norm_file).replace(".json", ".pkl"))  #
        with open(norm_file) as f:
            norm_res = json.load(f)
        pretrain_ins = []
        for section in norm_res:
            chunk_ins = []
            for block in norm_res[section]:
                for ins in block[1]:
                    ins_tokens = []
                    ins = ins.split(";")[3]
                    split_ins = ins.split("\t")
                    opcode = "_".join(split_ins[0].split())
                    ins_tokens.append(opcode)
                    if len(split_ins) > 1:
                        for operand in split_ins[1].split(","):
                            operand = operand.strip()
                            if operand:
                                ins_tokens.append(operand)
                    ins_tokens.append("<eos>")
                    if len(chunk_ins) + len(ins_tokens) > self.chunk_size:
                        chunk_str = " ".join(chunk_ins)
                        pretrain_ins.append(chunk_str)
                        chunk_ins = ins_tokens
                    else:
                        chunk_ins.extend(ins_tokens)

        with open(out_file, "wb") as f:
            pickle.dump(pretrain_ins, f)

    def batch_for_genpkl(self):
        json_files = []
        for json_flle in glob('{}/*.json'.format(self.norm_dir)):
            dest_path = os.path.join(self.pretrain_dir, os.path.basename(json_flle).replace(".json", ".pkl"))
            if not os.path.exists(dest_path):
                json_files.append(json_flle)
        with tqdm(total=len(json_files)) as progress_bar, ThreadPoolExecutor(max_workers=self.workers) as executor:
            for _ in executor.map(self.genpkl_perfile, json_files, ):
                progress_bar.update(1)

    def make_dataset_perthread(self, pickle_files, chunk_size, dataset_dir, lock, current_index):
        shared_content = []
        while True:
            with lock:
                if not pickle_files:
                    break
                pickle_file = pickle_files.pop(0)
            try:
                with open(pickle_file, "rb") as f:
                    pickle_content = pickle.load(f)

                remain_index = 0
                while remain_index < len(pickle_content):
                    space_left = chunk_size - len(shared_content)
                    end_index = min(remain_index + space_left, len(pickle_content))

                    shared_content.extend(pickle_content[remain_index:end_index])

                    if len(shared_content) == chunk_size:
                        with lock:
                            db_file_name = os.path.join(dataset_dir, f'dataset_{current_index[0]}.pkl')
                            current_index[0] += 1
                        with open(db_file_name, "wb") as db:
                            pickle.dump(shared_content, db)
                        shared_content = []

                    remain_index = end_index

            except Exception as e:
                print(f"Error processing file {pickle_file}: {e}")

        if shared_content:
            with lock:
                db_file_name = os.path.join(dataset_dir, f'dataset_{current_index[0]}.pkl')
                current_index[0] += 1
            with open(db_file_name, "wb") as db:
                pickle.dump(shared_content, db)

    def make_dataset(self, pickle_dir, dataset_dir, chunk_size, num_threads):
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)

        pickle_files = glob('{}/*.pkl'.format(pickle_dir))
        lock = threading.Lock()
        file_index = [0]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                executor.submit(self.make_dataset_perthread, pickle_files, chunk_size, dataset_dir, lock, file_index)


def replace(match):
    return "0x" + match.group(0)


def fix_palmtree_const(pklfile):
    with open(pklfile, "rb") as f:
        res = pickle.load(f)
    new_res = []
    pattern = r'\b\d+\b'
    for sample in res:
        text = sample[0]
        ins = text.split(",")
        ins_new = []
        label = sample[1]
        for i in ins:
            tokens = i.split()
            tokens_new = []
            skip = False
            brace = False
            for token in tokens:
                if token == "{" or token == "(":
                    skip = True
                    if token == "{":
                        brace = True
                    continue
                if token == "}" or token == ")":
                    skip = False
                    brace = False
                    continue
                if skip:
                    if not brace:
                        tokens_new[-1] = tokens_new[-1] + token  # st ( 7 ) -> st7
                    continue
                token_new = re.sub(pattern, replace, token)
                tokens_new.append(token_new)
            tokens_new_str = ' '.join(tokens_new)
            ins_new.append(tokens_new_str)
        text_new = ",".join(ins_new)
        new_res.append([text_new, label])
    with open(pklfile.replace(".pkl", "2.pkl"), "wb") as f:
        pickle.dump(new_res, f)
    with open(pklfile.replace(".pkl", "2.json"), "w") as f:
        json.dump(new_res, f)
    # f.write(json.dumps(new_res))


if __name__ == "__main__":
    asm_codes = [
        "lea\tesi, [esi]",
        "nop",
        "lea\tecx, [edx + 0x400000]",
        "call\t0x408b60"
    ]

    # for asm_code in asm_codes:
    #     print(parse_instruction(asm_code))

    #
    # palmtree_generator.gen_dataset("data")
    # palmtree_generator.merge_datasets("dataset/1w","dataset/1w/1w_merged.pkl")
    # palmtree_generator.merge_datasets_xda("dataset/xda/0.1w","dataset/xda/0.1w/0.1w_merged.data","dataset/xda/0.1w/0.1w_merged.label")
    # palmtree_generator.merge_datasets("dataset/1w","dataset/1w/1w_merged.pkl")
    # palmtree_generator.merge_datasets("dataset/5w","dataset/5w/5w_merged.pkl")

    with open("configs/Config.json") as f:
        configs = json.load(f)
    # ## Generate classify dataset
    # classify_generator = GenClassifyDataset(configs["data_num"], configs["label"], configs["chunk_size"],
    #                                         configs["sample_num"], configs["norm_dir"], configs["dataset_dir"],
    #                                         configs["drop_threshold"], configs["split_region"])
    # classify_generator.gen_dataset("le")

    # classify_generator.merge_datasets("dataset/deeppacker/1.0w","dataset/deeppacker/1.0w/2w_merged.pkl")
    # classify_generator.merge_datasets("dataset/palmtree/1.0w","dataset/palmtree/1.0w/2w_merged.pkl")
    # classify_generator.merge_datasets_xda("D:/finetune_dataset/dataset/xda/task1","D:/finetune_dataset/dataset/xda/task1")
    # classify_generator.merge_datasets("D:/finetune_dataset/dataset/palmtree/task1","D:/finetune_dataset/dataset/palmtree/task1")

    ## Generate pretrain dataset
    # pretrain_generator=GenPreTrainDataset(512, "D:/finetune_dataset/sourceforge_pretrain", "D:/pretrain_dataset")
    # pretrain_generator.batch_for_genpkl()
    # pretrain_generator.make_dataset("D:/pretrain_dataset","D:/pretrain",100000, 8)

    # norm_files=glob('{}/*.json'.format("D:/finetune_dataset/LPD_norm"))
    # random.shuffle(norm_files)
    # for i in range(2000):
    #     shutil.move(norm_files[i], "D:/finetune_dataset/LPD_finetune")

    # fix_palmtree_const(r"D:\finetune_dataset\dataset\palmtree\5.0w\data_5.0w.pkl")
    # fix_palmtree_const(r"D:\finetune_dataset\dataset\palmtree\5.0w\code_5.0w.pkl")
    # fix_palmtree_const(r"D:\finetune_dataset\dataset\palmtree\2.5w\le_2.5w.pkl")
    # fix_palmtree_const(r"D:\finetune_dataset\dataset\palmtree\2.5w\packed_2.5w.pkl")

    models = ["deeppacker", "palmtree", "xda"]
    tasks = ["task2", "task3"]
    datasets = ["train", "val", "test"]
    for model in models:
        for task in tasks:
            for dataset in datasets:
                dataset_path = f"D:/finetune_dataset/dataset/{model}/{task}/{dataset}.pkl"
                dest_path = f"D:/finetune_dataset/dataset/{model}/{task}/{dataset}2.pkl"
                with open(dataset_path, "rb") as f:
                    res = pickle.load(f)
                new_res = []
                for sample in res:
                    if sample[1] == 2:
                        if task == "task2":
                            new_res.append([sample[0], 1])
                        else:
                            new_res.append([sample[0], 0])
                    else:
                        new_res.append(sample)
                with open(dest_path, "wb") as f:
                    pickle.dump(new_res, f)

