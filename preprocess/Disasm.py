import idaapi, idc, idautils, ida_bytes
import json
from capstone import *
import time
import re
from wsgiref.util import request_uri
import copy
from tqdm import tqdm
from itertools import groupby
from LE import LowEntropyEncoding

win_skip_sections = ["DATA", "LOAD", ".idata", ".edata", ".bss", ".debug", ".drective", ".reloc", ".sbss", ".sdata",
                     ".srdata", ".sxdata", ".tls", ".vsdata"]
elf_skip_sections = [".got.plt", ".interp", ".gnu.hash", ".dynsym", ".dynstr", ".rel.dyn", ".plt.got", ".bss",
                     ".symtab", ".tbss"]

packer_sections = {
    "expressor": [".pdata"],
    "mpress": [".MPRESS1"],
    "aspack": [".text", "CODE"],
    "fsg": ["seg001", "seg002"],
    "upx": ["UPX1"],
    "winupack": [".rsrc"],
    "petite": ["seg000", "seg001"],
    "kkrunchy": ["kkrunchy"],
    "nspack": ["nsp1", "nsp2"],
    "themida3": [".boot", "________"],
    "acprotect": [".perplex", ".text"],
    "pecompact": [".text"]
}


class Disassembler():
    def __init__(self, json_output_file, configs):
        self.json_output_file = json_output_file
        self.configs = configs
        self.architecture = configs["architecture"]
        self.packer = configs["packer"]
        self.splitregion = configs["split_region"]
        self.LE = configs["LE"]
        self.next_ea = 0
        self.seg_list = {}
        self.disasm_res = {}
        self.current_type = None
        self.current_start = None
        self.md = None
        self.initialize_disassembler()

    def initialize_disassembler(self):
        idaapi.auto_wait()
        if idaapi.idainfo_is_32bit():
            self.md = Cs(CS_ARCH_X86, CS_MODE_32)
        elif idaapi.idainfo_is_64bit():
            self.md = Cs(CS_ARCH_X86, CS_MODE_64)
        self.md.detail = True

    def getSections(self):
        """
        format "SegmentName" : (start_ea, end_ea)
        """
        for n in range(idaapi.get_segm_qty()):
            seg = idaapi.getnseg(n)
            seg_name = idaapi.get_segm_name(seg)
            seg_attr = idc.get_segm_attr(seg.start_ea, idc.SEGATTR_PERM)
            if self.LE and self.packer:
                if seg_name not in packer_sections[self.packer]:
                    continue
            ## segment filter
            if not idaapi.is_loaded(seg.start_ea):
                ## current segment not have value
                continue
            if self.LE:
                if not (seg_attr & idaapi.SEGPERM_EXEC):
                    # only select executable segment
                    continue
            else:
                if (self.architecture == "ELF32" and seg_name in elf_skip_sections) or (
                        self.architecture == "PE32" and seg_name in win_skip_sections):
                    # skip specific segment
                    continue
                elif not (seg_attr & idaapi.SEGPERM_EXEC) and not (seg_attr & idaapi.SEGPERM_WRITE):
                    # skip read-only segment
                    continue
            if seg.end_ea > self.next_ea:
                self.next_ea = seg.end_ea
            self.seg_list.update({seg_name + "@@" + str(seg.start_ea): (seg.start_ea, seg.end_ea)})
            if self.splitregion:
                self.disasm_res.update({seg_name + "@@" + str(seg.start_ea): {
                    "seg_start": seg.start_ea,
                    "seg_end": seg.end_ea,
                    "ins_num": 0,
                    "region": []
                }})
            else:
                self.disasm_res.update({seg_name + "@@" + str(seg.start_ea): {
                    "seg_start": seg.start_ea,
                    "seg_end": seg.end_ea,
                    "ins_num": 0,
                    "ins": []
                }})

    def split_region(self):
        for seg_name in self.disasm_res:
            print(seg_name)
            start_ea, end_ea = self.seg_list[seg_name]
            current_type = None
            current_start = start_ea
            while start_ea < end_ea:
                flags = idaapi.get_flags(start_ea)
                if idaapi.is_code(flags):
                    ea_type = "code"
                else:
                    ea_type = "data"
                if current_type is None:
                    current_type = ea_type
                    current_start = start_ea
                elif ea_type != current_type:
                    # threshold=20
                    if current_type == "code" and idaapi.get_item_size(start_ea) <= 20:
                        pass
                    else:
                        self.disasm_res[seg_name]["region"].append(
                            [current_type, current_start, start_ea - 1, start_ea - current_start, []])
                        current_type = ea_type
                        current_start = start_ea

                start_ea = start_ea + idaapi.get_item_size(start_ea)
            if current_type is not None:
                self.disasm_res[seg_name]["region"].append(
                    [current_type, current_start, start_ea, start_ea - current_start, []])

    def LE_sections(self):
        sections = copy.deepcopy(self.seg_list)
        for seg_name in sections:
            start_ea = self.disasm_res[seg_name]["seg_start"]
            end_ea = self.disasm_res[seg_name]["seg_end"]
            ins_bytes = idaapi.get_bytes(start_ea, end_ea - start_ea)
            ins_bytes_dict = LowEntropyEncoding(ins_bytes).leresult
            for key in ins_bytes_dict.keys():
                self.next_ea += 1
                if key == "origin":
                    continue
                ins_bytes = ins_bytes_dict[key]
                ins_bytes_len = len(ins_bytes)
                if not ins_bytes:
                    continue
                LE_segname = seg_name + "@@" + key
                print(LE_segname)
                idaapi.add_segm(1, self.next_ea, self.next_ea + ins_bytes_len, LE_segname, "CODE")
                try:
                    idaapi.mem2base(ins_bytes, self.next_ea)
                except Exception as e:
                    print(e)
                idaapi.auto_wait()
                self.seg_list.update({LE_segname: (self.next_ea, self.next_ea + ins_bytes_len)})
                self.disasm_res.update({LE_segname: {
                    "seg_start": self.next_ea,
                    "seg_end": self.next_ea + ins_bytes_len,
                    "ins_num": 0,
                    "ins": []
                }})
                self.next_ea = self.next_ea + ins_bytes_len
        for seg_name in sections:
            del self.disasm_res[seg_name]
            del self.seg_list[seg_name]

    def Force2Code(self, ea):
        """
        check if current byte is data
        """
        if not idaapi.is_code(idaapi.get_flags(ea)):
            idaapi.del_items(ea, idaapi.DELIT_EXPAND)
            idaapi.auto_wait()
            idaapi.create_insn(ea)
            idaapi.auto_wait()
        if idaapi.get_item_size(ea):
            return ea + idaapi.get_item_size(ea)
        else:
            return ea + 1

    def LinearProcess(self):
        """
            Linear Disassembly
        """
        for seg_name in self.seg_list:
            start_ea, end_ea = self.seg_list[seg_name]
            while start_ea < end_ea:
                start_ea = self.Force2Code(start_ea)

    def getInstructions(self, start_ea, end_ea):
        ins_list = []
        ins_num = 0
        while start_ea <= end_ea:
            if idc.is_code(idc.get_full_flags(start_ea)):
                ins_bytes = idc.get_bytes(start_ea, idc.get_item_size(start_ea))
                try:
                    for i in self.md.disasm(ins_bytes, start_ea):
                        insn = "%s\t%s" % (i.mnemonic, i.op_str)
                        ins_bytes_hex = " ".join(["{:02x}".format(x) for x in i.bytes])
                        # ins_list.append("{};{};{};{}".format(head,ins_bytes_len,insn,ins_bytes_hex))
                        ins_list.append("{};{};{}".format(start_ea, insn, ins_bytes_hex))
                        ins_num += 1
                except:
                    pass
                start_ea += idc.get_item_size(start_ea)
            else:
                data_len = 0
                data_start = start_ea
                while start_ea <= end_ea and not idc.is_code(idc.get_full_flags(start_ea)):
                    data_len += idc.get_item_size(start_ea)
                    start_ea += idc.get_item_size(start_ea)
                bytes_data = idaapi.get_bytes(data_start, data_len)
                if bytes_data:
                    ins_list.append("data;{};{}".format(data_start, bytes_data.hex()))

        return ins_list, ins_num

    def insOutput(self):
        for seg_name in self.seg_list:
            if self.splitregion:
                regions = self.disasm_res[seg_name]["region"]
                for i in range(len(regions)):
                    start_ea = regions[i][1]
                    end_ea = regions[i][2]
                    ins_list, ins_num = self.getInstructions(start_ea, end_ea)
                    self.disasm_res[seg_name]["region"][i][4] = ins_list
                    self.disasm_res[seg_name]["ins_num"] += ins_num
            else:
                start_ea = self.disasm_res[seg_name]["seg_start"]
                end_ea = self.disasm_res[seg_name]["seg_end"]
                ins_list, ins_num = self.getInstructions(start_ea, end_ea)
                self.disasm_res[seg_name]["ins"] = ins_list
                self.disasm_res[seg_name]["ins_num"] = ins_num

    def merge_sections(self):
        sections = list(self.disasm_res.keys())
        for section in sections:
            if self.LE:
                section_name = section.split("@@")[0] + "@@" + section.split("@@")[2]
                if section_name not in self.disasm_res.keys():
                    self.disasm_res[section_name] = self.disasm_res[section]
                else:
                    self.disasm_res[section_name]["ins"].extend(self.disasm_res[section]["ins"])
                    self.disasm_res[section_name]["ins_num"] += self.disasm_res[section]["ins_num"]
            else:
                section_name = section.split("@@")[0]
                if section_name not in self.disasm_res.keys():
                    self.disasm_res[section_name] = self.disasm_res[section]
                else:
                    self.disasm_res[section_name]["region"].extend(self.disasm_res[section]["region"])
                    self.disasm_res[section_name]["ins_num"] += self.disasm_res[section]["ins_num"]
        for section in sections:
            del self.disasm_res[section]

    def run(self):
        idaapi.auto_wait()
        START_EA = idaapi.inf_get_min_ea()
        END_EA = idaapi.inf_get_max_ea()

        start = time.time()
        self.getSections()
        print("getsection:", time.time() - start)

        if self.LE:
            start_1 = time.time()
            self.LE_sections()
            print("LEsection:", time.time() - start_1)

        if self.splitregion:
            start_1 = time.time()
            self.split_region()
            print("splitregion:", time.time() - start_1)

        idaapi.auto_wait()
        start_2 = time.time()
        self.LinearProcess()
        print("LinearProcess:", time.time() - start_2)

        start_3 = time.time()
        self.insOutput()
        print("disasm_res:", time.time() - start_3)

        self.merge_sections()
        self.disasm_res["info"] = {"START_EA": START_EA, "END_EA": END_EA, "filepath": json_output_file[:-5]}
        try:
            with open(json_output_file, "w") as f:
                json.dump(self.disasm_res, f, indent=4)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    json_output_file = idc.ARGV[1]
    with open("configs/Config.json") as f:
        configs = json.load(f)
    disassembler = Disassembler(json_output_file, configs)
    disassembler.run()
    idc.qexit(0)