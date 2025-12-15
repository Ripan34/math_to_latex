class LatexTokenizer:
    def __init__(self):
        self.pad = "<pad>"
        self.bos = "<bos>"
        self.eos = "<eos>"

        self.special = [self.pad, self.bos, self.eos]

        self.symbols = [
            "\\frac", "\\sqrt", "{", "}", "^", "_",
            "+", "-", "=", "(", ")", "[", "]",
        ]

        self.id2tok = self.special + self.symbols
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}

    def encode(self, latex):
        seq = [self.tok2id[self.bos]]
        tokens = latex.replace("{", " { ").replace("}", " } ").split()
        for t in tokens:
            seq.append(self.tok2id.get(t, len(self.tok2id)))  # unknown added dynamically
        seq.append(self.tok2id[self.eos])
        return seq