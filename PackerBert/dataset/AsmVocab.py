class AsmVocab(object):
    def __init__(self, vocab: list=[]):
        self.vocab = ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"] + vocab
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4

        self.word2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2word = {idx: token for idx, token in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, idx):
        return self.vocab[idx]

    def get_id(self, token):
        if token in self.vocab:
            return self.word2id[token]
        else:
            return self.unk_index

    def save(self, vocab_path):
        with open(vocab_path, 'w') as f:
            f.write('\n'.join(self.vocab))

    def load(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocab = f.read().split('\n')
            self.pad_index = self.vocab.index('<pad>')
            self.unk_index = self.vocab.index('<unk>')
            self.eos_index = self.vocab.index('<eos>')
            self.sos_index = self.vocab.index('<sos>')
            self.mask_index = self.vocab.index('<mask>')

            self.word2id = {token: idx for idx, token in enumerate(self.vocab)}
            self.id2word = {idx: token for idx, token in enumerate(self.vocab)}


if __name__ == '__main__':
    xal = AsmVocab([])
    xal.load(r'vocab.txt')
    print(xal.word2id)
