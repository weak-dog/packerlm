import idaapi, idc, idautils, ida_bytes, ida_loader
import json
from capstone import *
import time
import re
from wsgiref.util import request_uri
import copy
from tqdm import tqdm
from itertools import groupby
from LE import LowEntropyEncoding
import pefile
import os

win_skip_sections = ["DATA", "LOAD", ".edata", ".bss", ".debug", ".drective", ".reloc", ".sbss", ".sdata", ".srdata",
                     ".sxdata", ".tls", ".vsdata"]  # ".idata"
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
    def __init__(self, json_output_file, sample_path, configs):
        self.json_output_file = json_output_file
        self.sample_path = sample_path
        self.static_path = self.sample_path.replace(".exe", ".static")
        self.configs = configs
        self.architecture = configs["architecture"]
        self.packer = configs["packer"]
        self.splitregion = configs["split_region"]
        self.LE = configs["LE"]
        self.overlay = configs["overlay"]
        self.static = configs["static"]
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

    def find_and_load_overlay(self):
        pe = pefile.PE(self.sample_path)
        file_size = os.path.getsize(self.sample_path)
        max_file_offset = 0
        max_end_va = 0
        for section in pe.sections:
            section_end_offset = section.PointerToRawData + section.SizeOfRawData
            max_file_offset = max(max_file_offset, section_end_offset)
        print(max_file_offset)
        for seg_ea in idautils.Segments():
            end_va = idc.get_segm_end(seg_ea)
            max_end_va = max(max_end_va, end_va)

        print(max_end_va)
        if file_size > max_file_offset:
            overlay_offset = max_file_offset
            overlay_size = file_size - overlay_offset

            print(hex(overlay_offset), hex(max_end_va), hex(overlay_size))
            with open(self.sample_path, "rb") as file:
                file.seek(overlay_offset)
                overlay_bytes = file.read(overlay_size)
            idaapi.add_segm(0, max_end_va, max_end_va + overlay_size, "overlay", "CODE")
            try:
                idaapi.mem2base(overlay_bytes, max_end_va)
            except Exception as e:
                print(e)
            idaapi.auto_wait()

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
            # if self.LE:
            # if not (seg_attr & idaapi.SEGPERM_EXEC):
            # # only select executable segment
            #     continue
            else:
                if (self.architecture == "ELF" and seg_name in elf_skip_sections) or (
                        self.architecture == "PE" and seg_name in win_skip_sections):
                    # skip specific segment
                    continue
                elif self.overlay and seg_name == "overlay":
                    pass
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
                    self.disasm_res[seg_name]["region"].append(
                        [current_type, current_start, start_ea - 1, start_ea - current_start, []])
                    current_type = ea_type
                    current_start = start_ea
                start_ea = start_ea + idaapi.get_item_size(start_ea)
            if current_type is not None:
                self.disasm_res[seg_name]["region"].append(
                    [current_type, current_start, start_ea - 1, start_ea - current_start, []])
        # 筛选太小的data
        for seg_name in self.disasm_res:
            # print(f"split:{seg_name}")
            regions = self.disasm_res[seg_name]["region"]
            # for region in regions:
            #     print(region[0], region[1])
            if len(regions) <= 2:
                continue
            regions_new = [regions[0]]
            # print("len:",len(regions))
            # print("!:",regions[0][0], regions[0][1], regions[0][3])
            i = 1
            merge = False
            while i < len(regions) - 1:
                # print(regions[i][0], regions[i][1], regions[i][3])
                if regions[i][0] == "code":
                    regions_new.append(regions[i])
                    i += 1
                    merge = False
                else:
                    if regions[i][3] < 10:
                        # print("dd:",regions[i][1])
                        regions_new[-1][2] = regions[i + 1][2]
                        regions_new[-1][3] += (regions[i][3] + regions[i + 1][3])
                        regions_new[-1][4] += (regions[i][4] + regions[i + 1][4])
                        i += 2
                        merge = True
                    else:
                        regions_new.append(regions[i])
                        i += 1
                        merge = False
            if regions[-1][0] == "data":
                regions_new.append(regions[-1])
            else:
                if not merge:
                    regions_new.append(regions[-1])

            self.disasm_res[seg_name]["region"] = regions_new

    def LE_sections(self):
        sections = copy.deepcopy(self.seg_list)
        for seg_name in sections:
            start_ea = self.disasm_res[seg_name]["seg_start"]
            end_ea = self.disasm_res[seg_name]["seg_end"]
            ins_bytes = idaapi.get_bytes(start_ea, end_ea - start_ea)

            # 这里需要过滤填充
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
                # print(LE_segname)
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

    def Force2Code_static(self, ea):

        """
        check if current byte is data
        """
        if not idaapi.is_code(idaapi.get_flags(ea)):
            start_head = ida_bytes.get_item_head(ea)
            idaapi.del_items(start_head, idaapi.DELIT_EXPAND)
            idaapi.auto_wait()
            next_ea = ea + 1
            # print("force ea:", hex(ea))
            while (not idaapi.create_insn(ea)):
                if next_ea - ea > 10:
                    # print("!!!break:", hex(ea), hex(next_ea))
                    break
                idaapi.del_items(next_ea, idaapi.DELIT_EXPAND)
                idaapi.auto_wait()
                next_ea += 1
            idaapi.auto_wait()
        else:
            # not start，
            # print(f"iscode:{ea}")
            pass
        if idaapi.get_item_size(ea):
            return ea + idaapi.get_item_size(ea)
        else:
            return ea + 1

    def Force2Code(self, ea):
        """
        check if current byte is data
        """
        if not idaapi.is_code(idaapi.get_flags(ea)):
            # start_head = ida_bytes.get_item_head(ea)
            idaapi.del_items(ea, idaapi.DELIT_EXPAND)
            idaapi.auto_wait()
            idaapi.create_insn(ea)
            idaapi.auto_wait()
        if idaapi.get_item_size(ea):
            start_head = ida_bytes.get_item_head(ea)
            if start_head != ea:
                idaapi.del_items(ea, idaapi.DELIT_EXPAND)
            return ea + idaapi.get_item_size(ea)
        else:
            return ea + 1

    def LinearProcess_static(self):
        with open(self.static_path) as f:
            lines = f.readlines()
        for line in lines:
            start_end_ea = line.split(",")[0]
            start_ea = int(start_end_ea.split("_")[0])
            end_ea = int(start_end_ea.split("_")[1])
            while start_ea <= end_ea:
                start_ea = self.Force2Code_static(start_ea)

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
                    if self.splitregion:
                        self.disasm_res[section_name]["region"].extend(self.disasm_res[section]["region"])
                        self.disasm_res[section_name]["ins_num"] += self.disasm_res[section]["ins_num"]
                    else:
                        self.disasm_res[section_name]["ins"].extend(self.disasm_res[section]["ins"])
                        self.disasm_res[section_name]["ins_num"] += self.disasm_res[section]["ins_num"]
        for section in sections:
            del self.disasm_res[section]

    def run(self):
        idaapi.auto_wait()
        START_EA = idaapi.inf_get_min_ea()
        END_EA = idaapi.inf_get_max_ea()

        if self.overlay:
            start = time.time()
            self.find_and_load_overlay()
            print("Load overlay:", time.time() - start)

        start = time.time()
        self.getSections()
        print("getsection:", time.time() - start)

        if self.static:
            idaapi.auto_wait()
            start_2 = time.time()
            self.LinearProcess_static()
            print("LinearProcess static:", time.time() - start_2)

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
    sample_path = idc.ARGV[2]

    with open("D:/DeepPacker/preprocess/configs/Config.json") as f:
        configs = json.load(f)
    disassembler = Disassembler(json_output_file, sample_path, configs)
    disassembler.run()
    idc.qexit(0)