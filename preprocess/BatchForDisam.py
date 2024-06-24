import os
import subprocess
import sys
import time
import signal
import psutil
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json

class BatchDisassembler:
    def __init__(self, configs):
        self.configs=configs
        self.disasm_dir = configs["disasm_dir"]
        self.ida_path = configs["ida_path"]
        self.ida_script = os.path.abspath(configs["ida_script"])
        self.samples_dir = configs["samples_dir"]
        self.timeout = configs["timeout"]
        self.log = configs["log"]
        self.threads = configs["ida_threads"]
        self.skip_finished=configs["skip_finished"]

    @staticmethod
    def file_type(filepath):
        try:
            with open(filepath, 'rb') as f:
                magic = f.read(2)
                return magic == b'\x7fE' or magic == b'\x4d\x5a'
        except:
            return False

    def process_file(self, sample_path):
        disasm_path=os.path.join(self.disasm_dir, os.path.basename(sample_path)+".json")
        ida_command = f'{self.ida_path} -c -A -S"{self.ida_script} {disasm_path}" {sample_path}'
        p = subprocess.Popen(ida_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start_time = time.time()
        try:
            while p.poll() is None:
                time.sleep(0.1)
                now = time.time()
                if now - start_time > self.timeout:
                    print("\ntimeout!")
                    with open(self.log, "a") as f:
                        f.write(sample_path + "\n")
                    os.kill(p.pid, signal.SIGTERM)
                    time.sleep(1)
                    if p.poll() is None:
                        try:
                            proc = psutil.Process(p.pid)
                            proc.terminate()
                        except Exception:
                            pass
                    break
        except Exception as e:
            print("\nError:", e)
            print(f"\n{ida_command}")

    def run(self):
        with open(self.log)as f:
            timeout_samples=[timeout_sample.strip() for timeout_sample in f.readlines()]
        sample_paths=[]
        for sample in tqdm(os.listdir(self.samples_dir)):
            sample_path=os.path.join(self.samples_dir,sample)
            if self.skip_finished:
                disasm_path=os.path.join(self.disasm_dir,sample+".json")
                if os.path.exists(disasm_path):
                    continue
            if self.file_type(sample_path):
                if sample_path not in timeout_samples:
                    sample_paths.append(sample_path)
        with tqdm(total=len(sample_paths))as progress_bar, ThreadPoolExecutor(max_workers=self.threads) as executor:
            for _ in executor.map(self.process_file, sample_paths, ):
                progress_bar.update(1)

if __name__ == "__main__":
    with open("configs/Config.json")as f:
        configs=json.load(f)
    disassembler = BatchDisassembler(configs)
    disassembler.run()
