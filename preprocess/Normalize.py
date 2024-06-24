import os
import re
import pickle
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from AsmVocab import AsmVocab
import json

x86_jump_opcodes = ["call","loop","loope","loopz","loopne","loopnz","jmp", "je", "jz", "jne", "jnz","ja", "jnbe", "jae", "jnb", "jb","jnae", "jbe", "jna", "jg", "jnle","jge", "jnl", "jl", "jnge", "jle","jng", "jc", "jnc", "jo", "jno","js", "jns", "jp", "jpe", "jnp", "jpo"]
remove_prefix=["dword","qword","oword","xword","yword","zword","xmmword","ymmword","zmmword","tbyte","word","byte","ptr","short","offset"]

# 
class Normalizer():
    def __init__(self, disasm_dir, norm_dir, split_region, workers=4):
        self.disasm_dir=disasm_dir
        self.norm_dir=norm_dir
        self.split_region=split_region
        self.workers = workers
        os.makedirs(self.norm_dir, exist_ok=True)

    def hex_to_dec(self, match, opcode, end_ea, mem):
        hex_value = match.group(0)
        dec_value = int(hex_value, 16)
        if opcode in x86_jump_opcodes:
            if dec_value > end_ea:
                return "mem-abnormal"
            else:
                return "mem-normal"
        else:
            if mem:
                if dec_value > end_ea:
                    if opcode=="lea":# vmprotect
                        return "const-normal"
                    else:
                        return "const-abnormal"
                else:
                    return "const-normal"
            else:
                return "const"
            
    def replace_hex_with_dec(self, assembly_code, opcode, end_ea, mem):
        def replacer(match):
            return self.hex_to_dec(match, opcode, end_ea, mem)
        pattern = r'0x[a-fA-F0-9]+'
        return re.sub(pattern, replacer, assembly_code)

    def normalize_opnd(self, opcode, opnd, end_ea, mem):
        # # st寄存器单独处理
        # if "st(" in opnd:
        #     return opnd
        dec_immediate_pattern = re.compile(r"\b[0-9a-fA-F]+\b")
        brace_pattern = re.compile(r'\{[^\}]*\}')
        opnd = brace_pattern.sub("", opnd)
        for prefix in remove_prefix:
            opnd = opnd.replace(prefix, "")
        opnd = opnd.replace("+", " _ ")
        opnd = opnd.replace("-", " _ ")
        opnd = opnd.replace("*", " * ")
        opnd = self.replace_hex_with_dec(opnd, opcode, end_ea, mem)
        opnd = dec_immediate_pattern.sub("const", opnd)
        opnd=opnd.replace(" ","")
        if opnd:
            if opnd[0]=="_":
                opnd=opnd[1:]
        return opnd.strip()

    def normalize_insn(self, asm, end_ea):
        asm_split=asm.split("\t")
        if len(asm_split)>1:
            try:
                opcode, operands = asm_split
            except:
                print("error",asm)
                return 0
            opcode=opcode.strip()
            opnd_strs = operands.split(", ")
            if opnd_strs:
                opnd_strs_norm=[]
                for opnd_str in opnd_strs:
                    # seg ptr
                    if ":" in opnd_str:
                        for opnd_split in opnd_str.split(":"):
                            mem = "[" in opnd_split
                            opnd_norm=self.normalize_opnd(opcode, opnd_split.strip(), end_ea, mem)
                            if opnd_norm:
                                opnd_strs_norm.append(opnd_norm)
                    else:
                        mem = "[" in opnd_str
                        opnd_norm = self.normalize_opnd(opcode, opnd_str.strip(), end_ea, mem)
                        if opnd_norm:
                            opnd_strs_norm.append(opnd_norm)
            return opcode, opnd_strs_norm
        else:
            opcode=asm.strip()
            opnd_strs=[]
            return opcode,opnd_strs
    
    def normalize_perfile(self, json_file):
        try:
            def normalize_line(line):
                nonlocal last_opcode
                line_res=""
                if "data;" in line:
                    line=line.strip()
                    data_bytes=line.split(";")[2]
                    for i in range(0, len(data_bytes), 2):
                        data_byte=data_bytes[i:i+2]
                        if data_byte not in ['00', 'ff', '90']:
                            line_res="pad-abnormal"
                            break
                        if i > 10:
                            break
                    if not line_res:
                        line_res="pad-normal"
                else:
                    ins=line.split(";")[1].strip()
                    try:
                        opcode,operands=self.normalize_insn(ins, end_ea)
                        operands_str = ', '.join(map(str, operands))
                        if last_opcode=="nop" or last_opcode=="int3":
                            if opcode!="nop" and opcode!="int3":
                                line_res=opcode+"\t"+operands_str
                        else:
                            line_res=opcode+"\t"+operands_str
                        last_opcode=opcode
                    except:
                        print("error:",ins)
                return line_res
            norm_res={}
            norm_path=os.path.join(self.norm_dir,os.path.basename(json_file))
            with open(json_file)as f:
                disasm_res=json.load(f)
            try:
                start_ea=disasm_res["info"]["START_EA"]
                end_ea=disasm_res["info"]["END_EA"]
            except:
                return 0
            for seg_name in disasm_res:
                if seg_name=="info":
                    continue
                norm_res[seg_name]=[]
                last_opcode=""
                if self.split_region:
                    regions=disasm_res[seg_name]["region"]
                    for i in range(len(regions)):
                        block=regions[i][4]
                        block_type=regions[i][0]
                        block_normal=[]
                        last_opcode=""
                        for line in block:
                            line_res=normalize_line(line)
                            if line_res:
                                block_normal.append(line+";"+line_res)
                        norm_res[seg_name].append([block_type,block_normal])
                else:
                    for line in disasm_res[seg_name]["ins"]:
                        line_res=normalize_line(line)
                        if line_res:
                            norm_res[seg_name].append(line+";"+line_res)
            with open(norm_path,"w")as f:
                f.write(json.dumps(norm_res))
        except Exception as e:
            print(e)
    
    def batch_for_normalize(self):
        json_files=[]
        for json_flle in glob('{}/*.json'.format(self.disasm_dir)):
            dest_path=os.path.join(self.norm_dir, os.path.basename(json_flle))
            if not os.path.exists(dest_path):
                json_files.append(json_flle)
        with tqdm(total=len(json_files)) as progress_bar, ThreadPoolExecutor(max_workers=self.workers) as executor:
            for _ in executor.map(self.normalize_perfile, json_files, ):
                progress_bar.update(1)

if __name__ == '__main__':
    with open("configs/Config.json")as f:
        configs=json.load(f)
    if configs["normalize"]:
        # TMP change for LPD
        # LPD_disasm_dir="D:/LPD_disasm_gt"
        # LPD_norm_dir="D:/LPD_norm_gt"
        # for packer in os.listdir(LPD_disasm_dir):
        #     normalizer = Normalizer(os.path.join(LPD_disasm_dir, packer), os.path.join(LPD_norm_dir, packer), configs["split_region"], configs["norm_threads"])
        #     normalizer.batch_for_normalize()
        normalizer = Normalizer(configs["disasm_dir"], configs["norm_dir"], configs["split_region"], configs["norm_threads"])
        normalizer.batch_for_normalize()

