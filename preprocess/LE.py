import random, base64, time, json
from random import randint
from math import log2
from collections import Counter
from itertools import cycle
import copy


class LowEntropyEncoding():
    def __init__(self, bytestring: bytes) -> None:
        self.bytestring = bytestring
        while self.bytestring[-1] == b"\x00" or self.bytestring[-1] == b"\xff":
            self.bytestring = self.bytestring.rstrip(b'\x00').rstrip(b'\xff')
        self.bslen = len(bytestring)
        self.ENTROPY_THRESHOLD = 7.0
        self.MAX_RETRY = 3
        self.leresult = {}
        self.encodingProcess()

    def encodingProcess(self):
        ins_bytes_hex = " ".join(["{:02x}".format(x) for x in self.bytestring])
        encodingresult = self.encoding()
        self.leresult.update({"origin": ins_bytes_hex})

        # base encoding
        print("[+] entropy of encoding result: {}, {} ".format(encodingresult[0][1], encodingresult[1][1]))
        if encodingresult[0][1] < self.ENTROPY_THRESHOLD:
            self.leresult.update({"base64": encodingresult[0][0]})
        if encodingresult[1][1] < self.ENTROPY_THRESHOLD:
            self.leresult.update({"base32": encodingresult[1][0]})

        # monoalphabetic substitution
        subresult = self.substitutionBased("mono")
        print("[+] entropy of monoalphabetic substitution result: {} ".format(subresult[0][1]))
        retry_times = 0
        while subresult[0][1] > self.ENTROPY_THRESHOLD and retry_times < self.MAX_RETRY:
            subresult = self.substitutionBased("mono")
            retry_times += 1
            print("[+] update entropy of monoalphabetic substitution result: {}".format(subresult[0][1]))
        if subresult[0][1] > self.ENTROPY_THRESHOLD:
            self.leresult.update({"monoalphabetic": ""})
        else:
            self.leresult.update({"monoalphabetic": subresult[0][0]})

        # polyalphabetic substitution
        subresult = self.substitutionBased("poly")
        print("[+] entropy of polyalphabetic substitution result: {} ".format(subresult[0][1]))
        retry_times = 0
        while subresult[0][1] > self.ENTROPY_THRESHOLD and retry_times < self.MAX_RETRY:
            subresult = self.substitutionBased("poly")
            retry_times += 1
            print("[+] update entropy of polyalphabetic substitution result: {}".format(subresult[0][1]))
        if subresult[0][1] > self.ENTROPY_THRESHOLD:
            self.leresult.update({"polyalphabetic": ""})
        else:
            self.leresult.update({"polyalphabetic": subresult[0][0]})

        # transpotition
        transresult = self.transpotition()
        retry_times = 0
        print("[+] entropy of transpotition result: {}, {} ".format(transresult[0][1], transresult[1][1]))
        while transresult[0][1] > self.ENTROPY_THRESHOLD and retry_times < self.MAX_RETRY:
            transresult = self.transpotition()
            retry_times += 1
        if transresult[0][1] > self.ENTROPY_THRESHOLD:
            self.leresult.update({"transpotation": ""})
        else:
            self.leresult.update({"transpotation": transresult[0][0]})

    def entropy(self, inputstring: bytes):
        """
            Calculate entropy
        """
        p, lns = Counter(inputstring), float(len(inputstring))
        return log2(lns) - sum(count * log2(count) for count in p.values()) / lns

    def encoding(self):
        """
            base64 and base32 encoding schemes
        """
        base64string = base64.b64encode(self.bytestring)
        base32string = base64.b32encode(self.bytestring)
        # TODO customencoding
        customencoding = ""

        return [(base64string, self.entropy(base64string)), (base32string, self.entropy(base32string))]

    def substitutionBased(self, mode, maxkeylength=4):
        """
            Monoalphabetic and Polyalphabetic substitution
            :param bytestring: input bytecode of instructions
            :param maxkeylength: max key length
            :param mode: mode of substitution
        """

        if mode == "mono":
            # Monoalphabetic xor substitution
            xorkey = randint(1, 255)
            monoXorString = bytes(a ^ xorkey for a in self.bytestring)
            return [(monoXorString, self.entropy(monoXorString))]
        else:
            # Polyalphabetic xor substitution
            xorkey = bytes(randint(1, 255) for i in range(maxkeylength))
            polyXorString = bytes(a ^ b for a, b in zip(self.bytestring, cycle(xorkey)))
            return [(polyXorString, self.entropy(polyXorString))]

    def transpotition(self, builtinshuffle=False):
        """
            shuffle bytes to recompose the original code.
            :param bytestring: input bytecode of instructions
            :return: a list of [transpoted string, shuffled array]
        """
        shuffledArray = []
        if builtinshuffle:
            result = "".join(random.sample(self.bytestring, self.bslen))
        else:
            result = bytearray(self.bytestring)
            shuffledtimes = int(self.bslen / 2)
            for i in range(shuffledtimes):
                origin = randint(0, self.bslen - 1)
                target = randint(0, self.bslen - 1)
                shuffledArray.append((origin, target))
                tmp = result[origin]
                result[origin] = result[target]
                result[target] = tmp
            result = bytes(result)
        return [(result, self.entropy(result)), shuffledArray]